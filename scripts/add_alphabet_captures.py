"""
Suma al dataset del abecedario las muestras grabadas desde la demo web.

La demo (signa-web, «Tu cámara te corrige» con ?captura en la dirección) graba
los mismos 266 valores que usa el modelo: 258 de la mano y 8 de la posición
respecto de la cara. Este script los agrega a alphabet_dataset.npz como fuentes
nuevas, una por archivo, para que la validación cruzada agrupada deje afuera
grabaciones enteras igual que deja afuera videos enteros.

Uso:
    python scripts/add_alphabet_captures.py muestras-*.json
    python scripts/train_alphabet.py && python scripts/calibrate_alphabet.py

El dataset original se guarda una sola vez como alphabet_dataset_base.npz y
siempre se parte de él, así correr el script de nuevo no duplica muestras.
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.hand_features import FACE_BLOCK_DIM, FEATURE_DIM  # noqa: E402

DATASET = ROOT / "data" / "processed_alphabet" / "alphabet_dataset.npz"
BASE = DATASET.with_name("alphabet_dataset_base.npz")
FIRST_SOURCE = 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--every", type=int, default=4,
                    help="tomar 1 de cada N cuadros (los cuadros seguidos son casi iguales)")
    args = ap.parse_args()

    if not BASE.exists():
        shutil.copy(DATASET, BASE)
    d = dict(np.load(BASE, allow_pickle=True))
    letters = [str(l) for l in d["letters"]]

    X, face, y, src, aug, mir = [], [], [], [], [], []
    added = Counter()
    for k, path in enumerate(args.files):
        data = json.loads(path.read_text())
        samples = data["samples"][:: args.every]
        for s in samples:
            f = np.asarray(s["features"], dtype=np.float32)
            if f.shape != (FEATURE_DIM + FACE_BLOCK_DIM,) or s["letter"] not in letters:
                continue
            X.append(f[:FEATURE_DIM])
            face.append(f[FEATURE_DIM:])
            y.append(letters.index(s["letter"]))
            src.append(FIRST_SOURCE + k)
            aug.append(False)
            mir.append(False)
            added[s["letter"]] += 1
        print(f"{path.name}: {len(samples)} muestras ({data.get('device', '?')[:60]})")

    if not X:
        sys.exit("No había muestras válidas.")
    d["X"] = np.vstack([d["X"], np.array(X)]).astype(d["X"].dtype)
    d["face"] = np.vstack([d["face"], np.array(face)]).astype(d["face"].dtype)
    d["y"] = np.concatenate([d["y"], np.array(y, dtype=d["y"].dtype)])
    d["sources"] = np.concatenate([d["sources"], np.array(src, dtype=d["sources"].dtype)])
    d["is_augmented"] = np.concatenate([d["is_augmented"], np.array(aug, dtype=d["is_augmented"].dtype)])
    d["mirrored"] = np.concatenate([d["mirrored"], np.array(mir, dtype=d["mirrored"].dtype)])
    np.savez(DATASET, **d)
    print("Agregadas por letra: " + ", ".join(f"{l} {n}" for l, n in sorted(added.items())))
    print(f"Dataset: {len(d['y'])} muestras → {DATASET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
