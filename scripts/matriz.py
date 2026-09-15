"""
Genera matriz de confusión y reporte de clasificación (precision/recall/f1
por seña) evaluando sobre el test set real — el mismo split que usa
train.py (mismo random_state=42), así que son secuencias que el modelo
nunca vio durante el entrenamiento.

Uso:
    python scripts/confusion_matrix.py
    python scripts/confusion_matrix.py --model models/saved_model/signa_model.keras

Salida:
    - Reporte de clasificación y top confusiones impresos en consola
    - reports/confusion_matrix.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.labels import load_config
from src.data.normalization import normalize_dataset
from scripts.train import load_dataset  # reutilizamos la misma función de carga que train.py

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "reports"


def main():
    parser = argparse.ArgumentParser(description="Matriz de confusión sobre el test set.")
    parser.add_argument(
        "--model", type=str,
        default=str(MODELS_DIR / "saved_model" / "signa_model.keras"),
        help="Ruta al modelo .keras entrenado",
    )
    args = parser.parse_args()

    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report

    config = load_config()
    training_cfg = config["training"]

    print("Cargando datos...")
    X, y, label_map = load_dataset(config)
    print(f"Dataset: {X.shape[0]} secuencias, {len(label_map)} señas")

    print("Normalizando...")
    X = normalize_dataset(X)

    num_classes = len(label_map)
    y_cat = tf.keras.utils.to_categorical(y, num_classes)

    test_size = training_cfg["test_size"]
    val_size = training_cfg["val_size"]

    # IMPORTANTE: mismo random_state=42 y mismo orden de splits que train.py,
    # para garantizar que evaluamos sobre el test set real (no visto en train).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size / (1 - test_size),
        random_state=42,
    )

    print(f"Test set: {len(X_test)} secuencias")
    print(f"\nCargando modelo desde: {args.model}")
    model = tf.keras.models.load_model(args.model)

    print("Prediciendo sobre test set...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    labels_sorted = [label_map[i] for i in range(num_classes)]

    # ---- Reporte de clasificación (precision/recall/f1 por clase) ----
    print("\n" + "=" * 60)
    print("REPORTE DE CLASIFICACIÓN")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred, target_names=labels_sorted, digits=3, zero_division=0
    )
    print(report)

    # ---- Matriz de confusión ----
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Top confusiones fuera de la diagonal: la forma más rápida de ver
    # qué pares de señas se mezclan más, sin tener que leer la matriz entera.
    print("=" * 60)
    print("TOP CONFUSIONES (real → predicho, cantidad de veces)")
    print("=" * 60)
    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], labels_sorted[i], labels_sorted[j]))
    confusions.sort(reverse=True)
    if confusions:
        for count, real, pred in confusions[:15]:
            print(f"  {real:<20} → {pred:<20}  ({count} veces)")
    else:
        print("  Sin confusiones — el modelo acertó todo en el test set.")

    # ---- Guardar imagen ----
    OUTPUT_DIR.mkdir(exist_ok=True)
    size = max(8, num_classes * 0.7)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión — test set")

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n✓ Matriz de confusión guardada en: {out_path}")


if __name__ == "__main__":
    main()