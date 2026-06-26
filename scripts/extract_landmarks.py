"""
Extrae landmarks desde videos crudos en data/raw/ y los guarda en data/processed/.

Si ya tenés .npy de collect_data.py, este script también sirve para
validar y mostrar estadísticas de lo que tenés grabado.

Uso:
    python scripts/extract_landmarks.py               # procesa todo data/raw/
    python scripts/extract_landmarks.py --sign hola   # solo una seña
    python scripts/extract_landmarks.py --stats        # muestra estadísticas de data/processed/
    python scripts/extract_landmarks.py --validate     # verifica shapes y NaNs en .npy existentes
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.extractor import get_holistic_model, extract_keypoints
from src.utils.labels import get_signs, load_config

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
EXPECTED_FEATURE_DIM = 258  # 132 pose + 63 lh + 63 rh


def process_video(video_path: Path, holistic, frames_per_sign: int) -> np.ndarray | None:
    """Extrae landmarks de un video y retorna array (frames, 258) o None si falla."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    sequence = []
    while len(sequence) < frames_per_sign:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

    cap.release()

    if len(sequence) < frames_per_sign:
        return None

    return np.array(sequence[:frames_per_sign])


def process_raw_videos(sign: dict, config: dict):
    """Procesa videos .mp4/.avi de data/raw/<folder>/ y los guarda en data/processed/."""
    frames_per_sign = config["collection"]["frames_per_sign"]
    raw_sign_dir = RAW_DIR / sign["folder"]
    processed_sign_dir = PROCESSED_DIR / sign["folder"]
    processed_sign_dir.mkdir(parents=True, exist_ok=True)

    if not raw_sign_dir.exists():
        print(f"  [SKIP] {sign['label']}: no hay carpeta en data/raw/")
        return 0

    video_files = list(raw_sign_dir.glob("*.mp4")) + list(raw_sign_dir.glob("*.avi"))
    if not video_files:
        print(f"  [SKIP] {sign['label']}: no hay videos en {raw_sign_dir}")
        return 0

    print(f"\nProcesando '{sign['label']}' — {len(video_files)} videos encontrados")
    saved = 0

    with get_holistic_model() as holistic:
        for i, video_path in enumerate(tqdm(video_files, desc=sign["label"])):
            out_path = processed_sign_dir / f"sequence_{i}.npy"
            if out_path.exists():
                continue  # no reprocesar
            seq = process_video(video_path, holistic, frames_per_sign)
            if seq is not None:
                np.save(out_path, seq)
                saved += 1
            else:
                print(f"  ✗ Video descartado (frames insuficientes): {video_path.name}")

    print(f"  ✓ {saved} secuencias guardadas en {processed_sign_dir}")
    return saved


def show_stats():
    """Muestra cuántas secuencias hay grabadas por seña."""
    signs = get_signs()
    print(f"\n{'Seña':<20} {'Secuencias':>10} {'Frames totales':>15}")
    print("-" * 48)
    total_seqs = 0
    for sign in signs:
        sign_dir = PROCESSED_DIR / sign["folder"]
        if not sign_dir.exists():
            print(f"{sign['label']:<20} {'0':>10} {'0':>15}")
            continue
        seqs = list(sign_dir.glob("sequence_*.npy"))
        frames = sum(np.load(s).shape[0] for s in seqs)
        print(f"{sign['label']:<20} {len(seqs):>10} {frames:>15}")
        total_seqs += len(seqs)
    print("-" * 48)
    print(f"{'TOTAL':<20} {total_seqs:>10}")


def validate_processed():
    """Verifica que los .npy tengan el shape correcto y no tengan NaNs."""
    signs = get_signs()
    config = load_config()
    frames_per_sign = config["collection"]["frames_per_sign"]
    issues = 0

    print(f"\nValidando data/processed/ (shape esperado: ({frames_per_sign}, {EXPECTED_FEATURE_DIM}))\n")

    for sign in signs:
        sign_dir = PROCESSED_DIR / sign["folder"]
        if not sign_dir.exists():
            continue
        for npy_file in sorted(sign_dir.glob("sequence_*.npy")):
            arr = np.load(npy_file)
            errors = []
            if arr.shape != (frames_per_sign, EXPECTED_FEATURE_DIM):
                errors.append(f"shape {arr.shape} (esperado ({frames_per_sign}, {EXPECTED_FEATURE_DIM}))")
            if np.isnan(arr).any():
                errors.append("contiene NaNs")
            if errors:
                print(f"  ✗ {sign['label']}/{npy_file.name}: {', '.join(errors)}")
                issues += 1

    if issues == 0:
        print("  ✓ Todo OK. No se encontraron problemas.")
    else:
        print(f"\n  {issues} archivos con problemas.")


def main():
    parser = argparse.ArgumentParser(description="Extrae landmarks desde videos crudos en data/raw/.")
    parser.add_argument("--sign", type=str, help="Procesa solo esta seña")
    parser.add_argument("--stats", action="store_true", help="Muestra estadísticas de lo grabado")
    parser.add_argument("--validate", action="store_true", help="Valida los .npy existentes")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.validate:
        validate_processed()
        return

    config = load_config()
    signs = get_signs()

    if args.sign:
        target = next((s for s in signs if s["label"] == args.sign), None)
        if not target:
            print(f"ERROR: Seña '{args.sign}' no encontrada.")
            sys.exit(1)
        signs = [target]

    total = 0
    for sign in signs:
        total += process_raw_videos(sign, config)

    print(f"\n✓ Extracción finalizada. {total} secuencias nuevas generadas.")
    print("Corré con --stats para ver el resumen, --validate para verificar integridad.")


if __name__ == "__main__":
    main()
