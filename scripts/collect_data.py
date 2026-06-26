"""
Graba secuencias de landmarks para las señas definidas en signs_config.yaml.

Uso:
    python scripts/collect_data.py
    python scripts/collect_data.py --sign hola --sequences 30
    python scripts/collect_data.py --list   # lista las señas disponibles

Salida:
    data/processed/<label>/sequence_<N>.npy
    Cada .npy es un array de shape (frames_per_sign, 258).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Aseguramos que src/ esté en el path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.extractor import get_holistic_model, process_frame
from src.utils.labels import get_signs, load_config

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def get_existing_sequences(sign_dir: Path) -> int:
    """Cuenta cuántas secuencias ya están grabadas para una seña."""
    return len(list(sign_dir.glob("sequence_*.npy")))


def collect_sign(sign: dict, config: dict, camera_index: int):
    frames_per_sign = config["collection"]["frames_per_sign"]
    sequences_to_record = config["collection"]["sequences_per_sign"]

    sign_dir = DATA_DIR / sign["folder"]
    sign_dir.mkdir(parents=True, exist_ok=True)

    existing = get_existing_sequences(sign_dir)
    start_seq = existing
    end_seq = existing + sequences_to_record

    print(f"\n{'='*50}")
    print(f"Seña: {sign['label'].upper()}")
    print(f"Secuencias existentes: {existing}")
    print(f"Grabando: {sequences_to_record} nuevas (total final: {end_seq})")
    print(f"Frames por secuencia: {frames_per_sign}")
    print(f"{'='*50}")
    print("Presioná Q en cualquier momento para cancelar.\n")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la cámara (índice {camera_index})")
        return

    with get_holistic_model() as holistic:
        for seq_num in range(start_seq, end_seq):
            # ---- Pantalla de espera entre secuencias ----
            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(
                    frame,
                    f"Seña: {sign['label']} | Secuencia {seq_num + 1}/{end_seq}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.putText(
                    frame, f"Listo en {countdown}...", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2,
                )
                cv2.imshow("Signa - Captura de datos", frame)
                if cv2.waitKey(1000) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Cancelado por el usuario.")
                    return

            # ---- Grabación de frames ----
            sequence = []
            for frame_num in range(frames_per_sign):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                annotated_frame, results, keypoints = process_frame(frame, holistic)

                cv2.putText(
                    annotated_frame,
                    f"GRABANDO | {sign['label']} | Sec {seq_num + 1}/{end_seq} | Frame {frame_num + 1}/{frames_per_sign}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )
                cv2.imshow("Signa - Captura de datos", annotated_frame)
                sequence.append(keypoints)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Cancelado por el usuario.")
                    return

            # ---- Guardar secuencia ----
            if len(sequence) == frames_per_sign:
                out_path = sign_dir / f"sequence_{seq_num}.npy"
                np.save(out_path, np.array(sequence))
                print(f"  ✓ Guardado: {out_path.name}")
            else:
                print(f"  ✗ Secuencia incompleta, descartada.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nListo. Secuencias grabadas en: {sign_dir}")


def main():
    parser = argparse.ArgumentParser(description="Graba secuencias de landmarks para señas LSA.")
    parser.add_argument("--sign", type=str, help="Graba solo esta seña (por label, ej: hola)")
    parser.add_argument("--sequences", type=int, help="Número de secuencias a grabar (override del config)")
    parser.add_argument("--camera", type=int, default=None, help="Índice de la cámara")
    parser.add_argument("--list", action="store_true", help="Lista las señas configuradas y sale")
    args = parser.parse_args()

    config = load_config()
    signs = get_signs()
    camera_index = args.camera if args.camera is not None else config["collection"]["camera_index"]

    if args.list:
        print("\nSeñas configuradas:")
        for s in signs:
            sign_dir = DATA_DIR / s["folder"]
            existing = get_existing_sequences(sign_dir) if sign_dir.exists() else 0
            print(f"  [{s['id']}] {s['label']:<15} — {existing} secuencias grabadas")
        return

    if args.sequences:
        config["collection"]["sequences_per_sign"] = args.sequences

    if args.sign:
        target = next((s for s in signs if s["label"] == args.sign), None)
        if not target:
            print(f"ERROR: Seña '{args.sign}' no encontrada en el config.")
            print(f"Señas disponibles: {[s['label'] for s in signs]}")
            sys.exit(1)
        collect_sign(target, config, camera_index)
    else:
        print(f"Grabando todas las señas ({len(signs)} en total)...")
        for sign in signs:
            collect_sign(sign, config, camera_index)

    print("\n✓ Captura finalizada.")


if __name__ == "__main__":
    main()
