"""
Exporta el modelo de señas dinámicas al formato que consume la app móvil.

La app NO usa el .tflite: corre la LSTM en JavaScript (ver
signa-mobile/src/features/ml/engine/signModel.ts), así que necesita los pesos
crudos más un manifiesto que diga en qué orden vienen.

Salida:
    weights.bin    float32 de todos los tensores, concatenados en el orden del
                   manifiesto, sin encabezado.
    manifest.json  nombre y shape de cada tensor + etiquetas + largo de ventana
                   + el umbral calibrado de cada seña (calibrate_signs.py).

Uso:
    python scripts/export_for_app.py
    python scripts/export_for_app.py --out ../signa-mobile/assets/models/lsa-signs-v3

Después de exportar hay que verificar la equivalencia contra Keras:
    cd ../signa-mobile && node scripts/verify-sign-model.mjs <manifest> <weights> <casos>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = ROOT / "models" / "saved_model" / "signa_model.keras"


def _ultimo(patron: str) -> Path:
    """El export más nuevo por número de versión, para no tener que tocar esto
    en cada reentrenamiento."""
    archivos = sorted(
        (ROOT / "models" / "exports").glob(patron),
        key=lambda f: int("".join(c for c in f.stem.split("_v")[-1] if c.isdigit()) or 0),
    )
    return archivos[-1] if archivos else ROOT / "models" / "exports" / patron.replace("*", "1")


DEFAULT_META = _ultimo("signa_model_v*_meta.json")
DEFAULT_OUT = ROOT.parent / "signa-mobile" / "assets" / "models" / "lsa-signs-v3"
DEFAULT_THRESHOLDS = _ultimo("signa_model_v*_thresholds.json")


def main() -> None:
    p = argparse.ArgumentParser(description="Exporta el modelo de señas para la app móvil.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--meta", type=Path, default=DEFAULT_META,
                   help="meta.json del .tflite, de donde salen las etiquetas")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS,
                   help="umbral por seña; sin esto la app usa uno global")
    args = p.parse_args()

    import tensorflow as tf

    modelo = tf.keras.models.load_model(args.model)

    # El orden importa: la app lee los tensores secuencialmente según este
    # manifiesto, así que tiene que ser el mismo al escribir y al leer.
    tensores, bloques = [], []
    for capa in modelo.layers:
        for w in capa.weights:
            a = np.asarray(w.numpy(), dtype=np.float32)
            tensores.append({"name": f"{capa.name}/{w.path.split('/')[-1]}",
                             "shape": list(a.shape)})
            bloques.append(a.ravel())

    pesos = np.concatenate(bloques)
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    etiquetas = [meta["labels"][str(i)] for i in range(len(meta["labels"]))]
    entrada = modelo.input_shape           # (None, 30, 258)

    umbrales = {}
    if args.thresholds.exists():
        umbrales = json.loads(args.thresholds.read_text(encoding="utf-8"))["thresholds"]
        faltan = [e for e in etiquetas if e not in umbrales]
        if faltan:
            print(f"  sin umbral (usan el de por defecto): {' '.join(faltan)}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "weights.bin").write_bytes(pesos.tobytes())
    (args.out / "manifest.json").write_text(json.dumps({
        "tensors": tensores,
        "labels": etiquetas,
        "sequenceLength": int(entrada[1]),
        "featureDim": int(entrada[2]),
        "thresholds": umbrales,
        # La app pone en cero el bloque de pose antes de inferir: el modelo se
        # entrena sin él para que no pueda clasificar por la postura del cuerpo.
        "poseIgnored": True,
    }, indent=2), encoding="utf-8")

    print(f"{len(pesos):,} parámetros · {len(pesos) * 4 / 1024:.0f} KB")
    print(f"  {args.out / 'weights.bin'}")
    print(f"  {args.out / 'manifest.json'}")
    print(f"  etiquetas: {' '.join(etiquetas)}")


if __name__ == "__main__":
    main()
