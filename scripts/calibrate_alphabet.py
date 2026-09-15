"""
Calibra un umbral por letra para el modo VERIFICACIÓN del abecedario.

El problema:
    En la escena "deletreá tu nombre" la app ya sabe qué letra pidió. Entonces
    no necesita resolver "¿cuál de las 26 es?" (identificación) sino "¿esto es
    una A?" (verificación). Es un problema bastante más fácil, y es el que
    permite que TODAS las letras sean usables — incluso las que en
    identificación pierden contra una vecina parecida.

    Ejemplo concreto: haciendo una Ñ, el modelo suele repartir probabilidad
    entre N y Ñ y a veces gana N. En identificación eso es un error. Pero si la
    app pidió una Ñ, sólo hay que mirar si p(Ñ) es lo bastante alta como para
    creerle — y ahí sí pasa.

Cómo:
    1. Validación cruzada agrupada por fuente → probabilidades out-of-fold
       (cada foto puntuada por un modelo que nunca la vio, ni a su señante).
    2. Para cada letra L, busca el umbral t_L que maximiza el F1 de
       "¿es esta foto una L?" usando p_L.
    3. Guarda los umbrales junto al .tflite.

    Un umbral por letra en vez de uno global importa porque las letras no
    tienen la misma confianza típica: Y sale con 99% y Q con 40%. Un umbral
    único o deja pasar cualquier cosa en Y o nunca acepta una Q.

Uso:
    python scripts/calibrate_alphabet.py
    python scripts/calibrate_alphabet.py --folds 5 --min-recall 0.85
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_alphabet import load_dataset, train_ensemble
from src.utils.thresholds import MIN_THRESHOLD, pick_threshold

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "alphabet_config.yaml"
DATASET_PATH = ROOT / "data" / "processed_alphabet" / "alphabet_dataset.npz"
EXPORTS_DIR = ROOT / "models" / "exports"
REPORTS_DIR = ROOT / "reports"


def out_of_fold_probs(X, y, sources, is_aug, letters, cfg, epochs, folds, seed, n_ensemble):
    """Probabilidades out-of-fold para cada foto real, agrupando por fuente."""
    uniq = np.array(sorted(set(sources.tolist())))
    chunks = np.array_split(np.random.default_rng(seed).permutation(uniq), min(folds, len(uniq)))

    probs = np.full((len(y), len(letters)), np.nan, dtype=np.float32)

    for k, held in enumerate(chunks, 1):
        in_held = np.isin(sources, held)
        train = ~in_held
        test = in_held & (is_aug == 0)
        if test.sum() == 0 or len(set(y[train].tolist())) < len(letters):
            print(f"  fold {k}: salteado")
            continue

        tr_idx = np.flatnonzero(train)
        rng = np.random.default_rng(seed + k)
        va_idx = rng.choice(tr_idx, size=max(1, len(tr_idx) // 10), replace=False)
        tr_mask = train.copy()
        tr_mask[va_idx] = False

        model = train_ensemble(X[tr_mask], y[tr_mask], X[va_idx], y[va_idx],
                               len(letters), cfg, epochs, n_ensemble, verbose=0)
        probs[test] = model.predict(X[test], verbose=0)
        acc = float((probs[test].argmax(1) == y[test]).mean())
        print(f"  fold {k}: fuentes {sorted(held.tolist())} → id {acc:.1%}")

    return probs


# Nunca bajamos de acá. Un umbral cerca de cero acepta cualquier mano y
# convierte la práctica en un placebo: la app dice "¡correcto!" hagas lo que
# hagas. Para una demo eso es peor que fallar, porque se nota enseguida.


def main():
    p = argparse.ArgumentParser(description="Calibra umbrales por letra.")
    p.add_argument("--dataset", type=Path, default=DATASET_PATH)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--ensemble", type=int, default=None)
    p.add_argument("--min-precision", type=float, default=0.40,
                   help="Precisión mínima por letra: evita umbrales que acepten cualquier cosa")
    p.add_argument("--min-recall", type=float, default=0.85,
                   help="Recall objetivo por letra (no se fuerza si rompe la precisión)")
    p.add_argument("--cache", type=Path, default=REPORTS_DIR / "alphabet_oof_probs.npz")
    p.add_argument("--recalibrate", action="store_true",
                   help="Reusar probabilidades cacheadas y sólo recalcular umbrales")
    p.add_argument("--seed", type=int, default=42)
    # Tienen que coincidir con los del entrenamiento del modelo que se calibra:
    # los umbrales por letra sólo valen para el espacio de features con el que
    # se midieron.
    p.add_argument("--with-face", action="store_true",
                   help="Sumar el bloque de posición respecto de la cara")
    p.add_argument("--face-repeat", type=int, default=32)
    args = p.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = config["training"]
    epochs = args.epochs or cfg["epochs"]
    n_ensemble = args.ensemble or cfg.get("ensemble", 1)

    X, y, sources, is_aug, letters = load_dataset(args.dataset, args.with_face,
                                                  args.face_repeat)
    print(f"\nDataset: {X.shape[0]} muestras · {len(letters)} letras\n")
    # La validación cruzada tarda varios minutos; los umbrales se re-derivan en
    # milisegundos. Cacheamos las probabilidades para poder retocar los pisos
    # sin volver a entrenar cinco ensembles.
    if args.recalibrate and args.cache.exists():
        cached = np.load(args.cache)
        probs, y = cached["probs"], cached["y"]
        print(f"Probabilidades cacheadas: {args.cache}")
    else:
        print("Probabilidades out-of-fold:")
        probs = out_of_fold_probs(X, y, sources, is_aug, letters, cfg, epochs,
                                  args.folds, args.seed, n_ensemble)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, probs=probs, y=y)
        print(f"  cacheadas en {args.cache} (--recalibrate las reusa)")

    scored = ~np.isnan(probs[:, 0])
    P, Y = probs[scored], y[scored]
    print(f"\nFotos puntuadas: {len(Y)}")
    print(f"Identificación (argmax entre {len(letters)}): {float((P.argmax(1) == Y).mean()):.1%}")

    thresholds, rows = {}, []
    for i, letter in enumerate(letters):
        is_letter = Y == i
        if is_letter.sum() == 0:
            thresholds[letter] = 0.5
            continue
        f1, thr, precision, recall = pick_threshold(
            P[:, i], is_letter, args.min_precision, args.min_recall)
        thresholds[letter] = round(thr, 3)
        id_acc = float((P[is_letter].argmax(1) == i).mean())
        rows.append((letter, thr, recall, precision, f1, id_acc, int(is_letter.sum())))

    rows.sort(key=lambda r: r[2])
    lines = [
        "Umbrales de verificación por letra (out-of-fold, agrupado por fuente)",
        f"Identificación global: {float((P.argmax(1) == Y).mean()):.1%}",
        "",
        "letra  umbral  recall  precisión   F1   ident.  n",
    ]
    for letter, thr, rec, prec, f1, id_acc, n in rows:
        lines.append(f"  {letter:>2}   {thr:5.2f}   {rec:5.0%}    {prec:5.0%}   {f1:.2f}  {id_acc:5.0%}  {n:>3}")

    worst = min(r[2] for r in rows)
    loose = [r[0] for r in rows if r[3] < args.min_precision]
    if loose:
        lines.append("")
        lines.append(f"Letras permisivas (precisión < {args.min_precision:.0%}): "
                     + ", ".join(loose)
                     + " — se aceptan fácil, pero también aceptan vecinas parecidas.")
    lines += [
        "",
        f"Recall más bajo entre las {len(rows)} letras: {worst:.0%}",
        "",
        "Cómo leerlo:",
        "  recall     = de las veces que hiciste esa letra, cuántas la acepta.",
        "               Es lo que se siente al usar la app en modo deletreo.",
        "  ident.     = cuántas veces gana entre las 26 sin saber qué se pidió.",
        "               Es el modo de 'Reconocimiento Simple'.",
        "  La brecha entre las dos columnas es exactamente lo que aporta",
        "  saber de antemano qué letra se pidió.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORTS_DIR / "alphabet_thresholds.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))

    # Se guarda al lado del .tflite más nuevo: el runner lo carga solo.
    models = sorted(EXPORTS_DIR.glob("signa_alphabet_v*.tflite"),
                    key=lambda p_: int("".join(c for c in p_.stem.split("_v")[-1] if c.isdigit()) or 0))
    if models:
        out = models[-1].with_name(models[-1].stem + "_thresholds.json")
        out.write_text(json.dumps({
            "thresholds": thresholds,
            "min_precision": args.min_precision,
            "min_recall": args.min_recall,
            "identification_accuracy": round(float((P.argmax(1) == Y).mean()), 4),
            "photos_scored": int(len(Y)),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nUmbrales: {out}")
    print(f"Reporte:  {report}")


if __name__ == "__main__":
    main()
