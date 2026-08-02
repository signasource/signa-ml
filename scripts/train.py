"""
Entrena un clasificador de señas LSA sobre los landmarks extraídos.

Flujo:
  1. Carga los .npy de data/processed/<seña>/sequence_N.npy
  2. Normaliza los keypoints (invariante a posición/distancia)
  3. Arma X (secuencias) e y (etiquetas numéricas)
  4. Split train/val/test
  5. Entrena una red densa simple (punto de partida, luego LSTM)
  6. Evalúa y guarda el modelo en models/saved_model/
  7. Exporta a TFLite en models/exports/

Uso:
    python scripts/train.py
    python scripts/train.py --epochs 30 --model dense
    python scripts/train.py --model lstm

Requiere entorno conda (signa) con tensorflow instalado.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.labels import get_signs, load_config
from src.data.normalization import normalize_dataset

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"


# ─── Carga de datos ──────────────────────────────────────────────────────────

def load_dataset(config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Carga todos los .npy de data/processed/ y devuelve X, y, label_map.

    Importante: los ids del config (signs_config.yaml) NO se usan directamente
    como etiquetas de entrenamiento, porque pueden no ser consecutivos
    (ej: si tenés ids 0, 1, 4 porque salteaste o agregaste señas, o si hay
    carpetas en data/processed/ que no figuran en el config). En cambio,
    se remapean a índices consecutivos 0..N-1 basados en lo que efectivamente
    se cargó. El mapeo final {nuevo_id: label} es el que se guarda en label_map
    y el que después va a la metadata del .tflite — así el modelo y el
    consumidor (signa-mobile) siempre están de acuerdo en qué índice es qué seña.

    Returns:
        X: (N, T, 258)
        y: (N,) etiquetas numéricas consecutivas (0..N_clases-1)
        label_map: {nuevo_id: label}
    """
    signs = get_signs()
    sequence_length = config["training"]["sequence_length"]
    known_folders = {sign["folder"] for sign in signs}

    # Carpetas en data/processed/ que no están en el config (ej: "hermanos")
    if DATA_DIR.exists():
        extra_dirs = [
            d.name for d in DATA_DIR.iterdir()
            if d.is_dir() and d.name not in known_folders and any(d.glob("sequence_*.npy"))
        ]
        if extra_dirs:
            print(f"  ⚠ Carpetas con datos que no están en signs_config.yaml: {extra_dirs}")
            print(f"    Agregalas al config si querés incluirlas en el entrenamiento.")

    X, y = [], []
    label_map = {}
    skipped = 0
    next_class_id = 0

    for sign in signs:
        sign_dir = DATA_DIR / sign["folder"]
        if not sign_dir.exists():
            print(f"  [SKIP] Sin datos para '{sign['label']}'")
            continue

        sequences = sorted(sign_dir.glob("sequence_*.npy"))
        if not sequences:
            print(f"  [SKIP] Sin secuencias para '{sign['label']}'")
            continue

        class_id = next_class_id
        label_map[class_id] = sign["label"]
        next_class_id += 1
        loaded = 0

        for seq_path in sequences:
            seq = np.load(seq_path)
            if seq.shape != (sequence_length, 258):
                skipped += 1
                continue
            X.append(seq)
            y.append(class_id)
            loaded += 1

        print(f"  ✓ {sign['label']}: {loaded} secuencias (clase {class_id})")

    if skipped:
        print(f"  ⚠ {skipped} secuencias descartadas por shape incorrecto")

    if not X:
        print("ERROR: No hay datos para entrenar. Corré collect_data.py primero.")
        sys.exit(1)

    if len(label_map) < 2:
        print("ERROR: Necesitás al menos 2 señas con datos para entrenar un clasificador.")
        sys.exit(1)

    return np.array(X), np.array(y), label_map


# ─── Modelos ─────────────────────────────────────────────────────────────────

def build_dense_model(sequence_length: int, feature_dim: int, num_classes: int):
    """
    Red densa simple. Buen punto de partida para pocas señas estáticas/simples.
    Input: (T, 258) aplanado → (T*258,)
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, feature_dim)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="signa_dense")

    return model


def build_lstm_model(sequence_length: int, feature_dim: int, num_classes: int):
    """
    LSTM bidireccional. Mejor para señas dinámicas con movimiento complejo.
    Captura dependencias temporales en la secuencia de landmarks.
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, feature_dim)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="signa_lstm")

    return model


# ─── Entrenamiento ────────────────────────────────────────────────────────────

def train(args):
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow no encontrado.")
        print("Activá el entorno conda: conda activate signa")
        sys.exit(1)

    config = load_config()
    training_cfg = config["training"]

    print("\nCargando datos...")
    X, y, label_map = load_dataset(config)
    print(f"\nDataset: {X.shape[0]} secuencias, {len(label_map)} señas")
    print(f"Shape X: {X.shape}  |  Shape y: {y.shape}")

    print("\nNormalizando...")
    X = normalize_dataset(X)

    # One-hot encoding
    num_classes = len(label_map)
    y_cat = tf.keras.utils.to_categorical(y, num_classes)

    # Split train / val / test
    from sklearn.model_selection import train_test_split
    test_size = training_cfg["test_size"]
    val_size = training_cfg["val_size"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size / (1 - test_size),
        random_state=42
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Construir modelo
    seq_len = training_cfg["sequence_length"]
    feat_dim = 258

    if args.model == "lstm":
        model = build_lstm_model(seq_len, feat_dim, num_classes)
    else:
        model = build_dense_model(seq_len, feat_dim, num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training_cfg["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    checkpoint_path = MODELS_DIR / "checkpoints" / "best_model.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        ),
    ]

    print(f"\nEntrenando modelo '{args.model}' por hasta {args.epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=training_cfg["batch_size"],
        callbacks=callbacks,
    )

    # Evaluación
    print("\nEvaluando en test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.2%}  |  Loss: {loss:.4f}")

    # Guardar modelo completo
    saved_model_path = MODELS_DIR / "saved_model" / "signa_model.keras"
    saved_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(saved_model_path))
    print(f"\nModelo guardado en: {saved_model_path}")

    # Exportar a TFLite
    export_tflite(model, label_map, accuracy)

    return history, accuracy


def export_tflite(model, label_map: dict, accuracy: float):
    import tensorflow as tf
    import tempfile
    import shutil
    import json

    exports_dir = MODELS_DIR / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    version = 1
    tflite_path = exports_dir / f"signa_model_v{version}.tflite"

    while tflite_path.exists():
        version += 1
        tflite_path = exports_dir / f"signa_model_v{version}.tflite"

    tmp_savedmodel = tempfile.mkdtemp()

    try:
        model.export(tmp_savedmodel)

        converter = tf.lite.TFLiteConverter.from_saved_model(
            tmp_savedmodel
        )

        tflite_model = converter.convert()

        tflite_path.write_bytes(tflite_model)

        print(
            f"TFLite exportado: {tflite_path} "
            f"({tflite_path.stat().st_size / 1024:.1f} KB)"
        )

    finally:
        shutil.rmtree(tmp_savedmodel, ignore_errors=True)

    meta = {
        "version": version,
        "accuracy": round(float(accuracy), 4),
        "labels": label_map,
        "model_name": model.name,
        "input_shape": list(model.input_shape[1:]),
    }

    meta_path = exports_dir / f"signa_model_v{version}_meta.json"

    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )

    print(f"Metadata guardada: {meta_path}")

# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entrena el clasificador de señas.")
    parser.add_argument(
        "--model", choices=["dense", "lstm"], default="dense",
        help="Arquitectura del modelo (default: dense)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Número máximo de epochs (default: usa el config)"
    )
    args = parser.parse_args()

    config = load_config()
    if args.epochs is None:
        args.epochs = config["training"]["epochs"]

    print(f"Modelo: {args.model} | Epochs máx: {args.epochs}")
    history, accuracy = train(args)
    print(f"\n✓ Entrenamiento finalizado. Accuracy final: {accuracy:.2%}")


if __name__ == "__main__":
    main()