"""
Entrena el clasificador de señas ESTÁTICAS (abecedario LSA).

Entrada:  data/processed_alphabet/alphabet_dataset.npz  (build_alphabet_dataset.py)
Salida:   models/exports/signa_alphabet_vX.tflite + _meta.json
          reports/alphabet_confusion.png, reports/alphabet_report.txt

Sobre el split (lo más importante de este script):
    Las fotos vienen de ~15 videos distintos y cada foto de una letra sale de
    un video distinto. Si partiéramos al azar, variantes aumentadas de la MISMA
    foto caerían en train y en test y la accuracy daría ~99% mintiendo.
    Por eso el default agrupa por FUENTE: el test set son videos que el modelo
    nunca vio. Es un número más bajo y mucho más parecido a lo que vas a ver
    frente a la cámara.

    Con tan pocas fuentes, un solo split es ruidoso. --cv corre
    leave-sources-out sobre todas las fuentes y reporta media ± desvío: ese es
    el número para confiar / poner en la presentación.

El escalado de features va DENTRO del modelo (capa Normalization adaptada al
train), así el .tflite recibe los 258 valores crudos y no hay que portar
ningún scaler a la app.

Uso:
    python scripts/train_alphabet.py
    python scripts/train_alphabet.py --cv              # validación cruzada por fuente
    python scripts/train_alphabet.py --split random    # sanity check (accuracy inflada)
    python scripts/train_alphabet.py --epochs 500 --no-export
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hand_features import FEATURE_DIM

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "alphabet_config.yaml"
DATASET_PATH = ROOT / "data" / "processed_alphabet" / "alphabet_dataset.npz"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


# ─── Datos ───────────────────────────────────────────────────────────────────

def load_dataset(path: Path, with_face: bool = False, face_repeat: int = 32):
    if not path.exists():
        print(f"ERROR: no existe {path}")
        print("Corré primero: python scripts/build_alphabet_dataset.py")
        sys.exit(1)

    d = np.load(path, allow_pickle=True)
    X, y, sources, is_aug = d["X"], d["y"], d["sources"], d["is_augmented"]
    letters = list(d["letters"])

    # Bloque opcional: dónde está la mano respecto de la cara. Va aparte en el
    # .npz para poder entrenar con y sin él sobre exactamente los mismos datos.
    if with_face:
        if "face" not in d:
            print("ERROR: el dataset no tiene el bloque de cara. "
                  "Reconstruilo: python scripts/build_alphabet_dataset.py")
            sys.exit(1)
        # El bloque se repite `face_repeat` veces a propósito. Son 8 valores
        # contra 258 de mano: puestos una sola vez, la red no los usa para nada
        # (medido: T→I sigue fallando igual). Repetidos hasta pesar parecido a
        # la mano, sí los usa. Con 32 el bloque ocupa 256 de 514 entradas, que
        # es donde deja de mejorar.
        X = np.hstack([X] + [d["face"]] * face_repeat).astype(np.float32)
        print(f"Bloque de posición respecto de la cara: {d['face'].shape[1]} valores "
              f"× {face_repeat} → {X.shape[1]} features totales")

    # Las letras sin ninguna foto se caen del modelo y los ids se recompactan,
    # para que el índice de salida del .tflite siempre sea 0..N-1 sin huecos.
    present = sorted(set(y.tolist()))
    remap = {old: new for new, old in enumerate(present)}
    kept = [letters[i] for i in present]
    y = np.array([remap[v] for v in y.tolist()], dtype=np.int64)

    if len(kept) < len(letters):
        dropped = [letters[i] for i in range(len(letters)) if i not in remap]
        print(f"⚠ Letras sin datos, excluidas del modelo: {', '.join(dropped)}")

    return X, y, sources, is_aug, kept


def group_split(sources, is_aug, y, n_test, n_val, seed):
    """
    Reserva fuentes enteras para val y test. Las muestras AUMENTADAS sólo van a
    train: val y test se evalúan contra fotos reales, no contra copias.
    Reintenta con distintas semillas hasta que train cubra todas las letras.
    """
    uniq = np.array(sorted(set(sources.tolist())))
    n_test = min(n_test, max(1, len(uniq) - 3))
    n_val = min(n_val, max(1, len(uniq) - n_test - 2))

    for attempt in range(200):
        rng = np.random.default_rng(seed + attempt)
        shuffled = rng.permutation(uniq)
        test_src = set(shuffled[:n_test].tolist())
        val_src = set(shuffled[n_test:n_test + n_val].tolist())

        in_test = np.isin(sources, list(test_src))
        in_val = np.isin(sources, list(val_src))
        train = ~(in_test | in_val)

        if len(set(y[train].tolist())) == len(set(y.tolist())):
            real = is_aug == 0
            return train, in_val & real, in_test & real, sorted(val_src), sorted(test_src)

    raise RuntimeError(
        "No se encontró un split por fuente que deje todas las letras en train. "
        "Bajá --test-sources / --val-sources, o usá --split random."
    )


def random_split(y, is_aug, seed):
    """Split estratificado clásico. Sirve de sanity check, NO de métrica real."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
    va, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=y[rest])
    mask = lambda ids: np.isin(idx, ids)
    real = is_aug == 0
    return mask(tr), mask(va) & real, mask(te) & real, [], []


# ─── Modelo ──────────────────────────────────────────────────────────────────

def build_ensemble(models):
    """
    Envuelve N redes ya entrenadas en UN solo modelo que promedia sus salidas.

    Se exporta como un único .tflite con la misma interfaz (258 entradas →
    N_letras salidas): la app no se entera de que adentro hay varias redes.
    Promediar redes entrenadas con semillas distintas corrige errores que cada
    una comete por su cuenta; con este dataset chico da ~2 puntos y, sobre
    todo, resultados más estables entre fuentes.
    """
    import tensorflow as tf

    if len(models) == 1:
        return models[0]

    inp = tf.keras.Input(shape=(models[0].input_shape[1],), name="hand_features")
    avg = tf.keras.layers.Average(name="letra")([m(inp) for m in models])
    return tf.keras.Model(inp, avg, name="signa_alphabet_ensemble")


def train_ensemble(X_tr, y_tr, X_va, y_va, num_classes, cfg, epochs, n, verbose=1):
    """Entrena n redes con semillas distintas y devuelve el modelo promediado."""
    import tensorflow as tf

    models = []
    for i in range(n):
        tf.keras.utils.set_random_seed(1000 + i)
        if verbose and n > 1:
            print(f"\n── red {i + 1}/{n} ──")
        # Keras 3 exige nombres únicos al anidar los modelos en el ensemble.
        m = build_model(X_tr, num_classes, cfg, name=f"signa_alphabet_{i}")
        fit(m, X_tr, y_tr, X_va, y_va, cfg, epochs, verbose=verbose)
        models.append(m)
    return build_ensemble(models)


def build_model(X_train, num_classes, cfg, name="signa_alphabet"):
    import tensorflow as tf

    # La entrada puede traer o no el bloque de posición respecto de la cara.
    n_features = int(X_train.shape[1])

    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(X_train)

    layers = [
        tf.keras.layers.Input(shape=(n_features,), name="hand_features"),
        norm,
        # El ruido sólo actúa en training: imita el jitter de MediaPipe frame a
        # frame, que es exactamente el ruido que el modelo va a ver en vivo.
        tf.keras.layers.GaussianNoise(cfg["landmark_noise_std"]),
    ]
    for units in cfg["hidden_units"]:
        layers += [
            tf.keras.layers.Dense(units, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(cfg["dropout"]),
        ]
    layers.append(tf.keras.layers.Dense(num_classes, activation="softmax", name="letra"))

    return tf.keras.Sequential(layers, name=name)


def fit(model, X_tr, y_tr, X_va, y_va, cfg, epochs, verbose=1):
    import tensorflow as tf

    # Las letras no tienen la misma cantidad de fotos; sin pesos el modelo
    # abandona las minoritarias, que suelen ser justo las difíciles.
    counts = np.bincount(y_tr, minlength=int(y_tr.max()) + 1).astype(np.float64)
    weights = {i: float(len(y_tr) / (len(counts) * c)) for i, c in enumerate(counts) if c > 0}

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            label_smoothing=cfg["label_smoothing"]
        ) if _supports_smoothing() else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=40,
                                         restore_best_weights=True, verbose=verbose),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=15, min_lr=1e-5, verbose=verbose),
    ]
    return model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs, batch_size=cfg["batch_size"],
        class_weight=weights, callbacks=callbacks, verbose=verbose,
    )


def _supports_smoothing() -> bool:
    """label_smoothing en SparseCategoricalCrossentropy no existe en todas las versiones."""
    import inspect
    import tensorflow as tf
    return "label_smoothing" in inspect.signature(
        tf.keras.losses.SparseCategoricalCrossentropy.__init__
    ).parameters


# ─── Reportes ────────────────────────────────────────────────────────────────

def write_reports(model, X_te, y_te, letters, extra_lines):
    from sklearn.metrics import classification_report, confusion_matrix

    probs = model.predict(X_te, verbose=0)
    pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_te, pred, labels=range(len(letters)))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = list(extra_lines)
    lines.append(classification_report(y_te, pred, labels=range(len(letters)),
                                       target_names=letters, zero_division=0))

    # Los pares que más se confunden: es la lista accionable para saber qué
    # letras conviene refotografiar o cuáles evitar en el video.
    pairs = []
    for i in range(len(letters)):
        for j in range(len(letters)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), letters[i], letters[j]))
    pairs.sort(reverse=True)
    if pairs:
        lines.append("\nConfusiones más frecuentes (real → predicho):")
        for n, real, wrong in pairs[:15]:
            lines.append(f"  {real} → {wrong}: {n}")

    report_path = REPORTS_DIR / "alphabet_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReporte: {report_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(8, len(letters) * 0.45),) * 2)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(letters)), letters, fontsize=8)
        ax.set_yticks(range(len(letters)), letters, fontsize=8)
        ax.set_xlabel("predicho")
        ax.set_ylabel("real")
        ax.set_title("Abecedario LSA — matriz de confusión (test)")
        fig.tight_layout()
        out = REPORTS_DIR / "alphabet_confusion.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Matriz:  {out}")
    except ImportError:
        pass

    return float((pred == y_te).mean())


# ─── Export ──────────────────────────────────────────────────────────────────

def export_tflite(model, letters, accuracy, cv_summary, n_ensemble):
    import tensorflow as tf

    exports = MODELS_DIR / "exports"
    exports.mkdir(parents=True, exist_ok=True)

    version = 1
    while (exports / f"signa_alphabet_v{version}.tflite").exists():
        version += 1
    path = exports / f"signa_alphabet_v{version}.tflite"

    tmp = tempfile.mkdtemp()
    try:
        model.export(tmp)
        tflite = tf.lite.TFLiteConverter.from_saved_model(tmp).convert()
        path.write_bytes(tflite)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    meta = {
        "version": version,
        "kind": "static_alphabet",
        "accuracy": round(accuracy, 4),
        "cv": cv_summary,
        "ensemble": n_ensemble,
        "labels": {str(i): l for i, l in enumerate(letters)},
        "input_shape": [int(model.input_shape[1])],
        "feature_spec": "src.data.hand_features.build_features (MediaPipe Hands, 258 valores)",
        "model_name": model.name,
    }
    meta_path = path.with_name(path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nTFLite:   {path} ({path.stat().st_size / 1024:.1f} KB)")
    print(f"Metadata: {meta_path}")


# ─── Validación cruzada ──────────────────────────────────────────────────────

def cross_validate(X, y, sources, is_aug, letters, cfg, epochs, folds, seed, n_ensemble):
    """Leave-sources-out: la estimación honesta de cuánto generaliza a gente nueva."""
    uniq = np.array(sorted(set(sources.tolist())))
    rng = np.random.default_rng(seed)
    chunks = np.array_split(rng.permutation(uniq), min(folds, len(uniq)))

    accs = []
    for k, held in enumerate(chunks, 1):
        in_held = np.isin(sources, held)
        real = is_aug == 0
        train = ~in_held
        test = in_held & real

        if len(set(y[train].tolist())) < len(letters) or test.sum() == 0:
            print(f"  fold {k}: salteado (train no cubre todas las letras)")
            continue

        # Un pedazo del train hace de validación para el early stopping.
        tr_idx = np.flatnonzero(train)
        rng_f = np.random.default_rng(seed + k)
        va_idx = rng_f.choice(tr_idx, size=max(1, len(tr_idx) // 10), replace=False)
        tr_mask = train.copy()
        tr_mask[va_idx] = False

        model = train_ensemble(X[tr_mask], y[tr_mask], X[va_idx], y[va_idx],
                               len(letters), cfg, epochs, n_ensemble, verbose=0)
        acc = float((model.predict(X[test], verbose=0).argmax(1) == y[test]).mean())
        accs.append(acc)
        print(f"  fold {k}: fuentes {sorted(held.tolist())} → {acc:.1%} ({int(test.sum())} fotos)")

    if not accs:
        return None
    summary = {"folds": len(accs), "mean": round(float(np.mean(accs)), 4),
               "std": round(float(np.std(accs)), 4),
               "per_fold": [round(a, 4) for a in accs]}
    print(f"\n  CV por fuente: {summary['mean']:.1%} ± {summary['std']:.1%} "
          f"({summary['folds']} folds)")
    return summary


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Entrena el clasificador del abecedario LSA.")
    p.add_argument("--dataset", type=Path, default=DATASET_PATH)
    p.add_argument("--split", choices=["group", "random"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--test-sources", type=int, default=None)
    p.add_argument("--val-sources", type=int, default=None)
    p.add_argument("--cv", action="store_true", help="Validación cruzada leave-sources-out")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--ensemble", type=int, default=None,
                   help="Redes a promediar (default: el config). 1 = modelo simple")
    p.add_argument("--with-face", action="store_true",
                   help="Sumar las 8 features de posición de la mano respecto de la cara")
    p.add_argument("--face-repeat", type=int, default=32,
                   help="Cuántas veces se repite ese bloque en la entrada (default 32)")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = config["training"]
    split_mode = args.split or cfg["split"]
    epochs = args.epochs or cfg["epochs"]
    n_ensemble = args.ensemble or cfg.get("ensemble", 1)

    X, y, sources, is_aug, letters = load_dataset(args.dataset, args.with_face,
                                                  args.face_repeat)
    print(f"\nDataset: {X.shape[0]} muestras · {len(letters)} letras · "
          f"{len(set(sources.tolist()))} fuentes")

    cv_summary = None
    if args.cv:
        print(f"\nValidación cruzada por fuente ({n_ensemble} redes por fold, tarda):")
        cv_summary = cross_validate(X, y, sources, is_aug, letters, cfg,
                                    epochs, args.cv_folds, args.seed, n_ensemble)

    if split_mode == "group":
        tr, va, te, val_src, test_src = group_split(
            sources, is_aug, y,
            args.test_sources or cfg["test_sources"],
            args.val_sources or cfg["val_sources"],
            args.seed,
        )
        print(f"\nSplit por fuente · val: {val_src} · test: {test_src}")
    else:
        tr, va, te, val_src, test_src = random_split(y, is_aug, args.seed)
        print("\nSplit aleatorio — OJO: esta accuracy está inflada, es sólo un sanity check.")

    print(f"Train: {int(tr.sum())}  |  Val: {int(va.sum())}  |  Test: {int(te.sum())} (fotos reales)")

    model = train_ensemble(X[tr], y[tr], X[va], y[va], len(letters), cfg,
                           epochs, n_ensemble)

    header = [
        f"Split: {split_mode}",
        f"Fuentes de test: {test_src}" if test_src else "",
        f"CV por fuente: {cv_summary['mean']:.1%} ± {cv_summary['std']:.1%}" if cv_summary else "",
        "",
    ]
    accuracy = write_reports(model, X[te], y[te], letters, [h for h in header if h != ""] + [""])
    print(f"\nTest accuracy: {accuracy:.1%}")

    saved = MODELS_DIR / "saved_model" / "signa_alphabet.keras"
    saved.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(saved))
    print(f"Modelo:   {saved}")

    if not args.no_export:
        export_tflite(model, letters, accuracy, cv_summary, n_ensemble)


if __name__ == "__main__":
    main()
