"""
Exploración interactiva de MediaPipe Holistic en tiempo real (versión matplotlib).

Usa matplotlib para mostrar el frame en vivo en lugar de cv2.imshow,
para evitar problemas de Qt/Wayland con OpenCV.

Muestra en pantalla:
  - Landmarks de pose (cuerpo completo)
  - Landmarks de manos (izquierda y derecha)
  - Estado de detección (pose / mano izq / mano der)

Uso:
    python notebooks/explore_mediapipe.py
    python notebooks/explore_mediapipe.py --camera 1   # cámara alternativa

Para salir: cerrá la ventana de matplotlib o Ctrl+C en la terminal.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.extractor import extract_keypoints, get_holistic_model, draw_landmarks


def main():
    parser = argparse.ArgumentParser(description="Exploración de MediaPipe Holistic en tiempo real.")
    parser.add_argument("--camera", type=int, default=0, help="Índice de cámara")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la cámara {args.camera}")
        sys.exit(1)

    print("MediaPipe Holistic — Exploración en tiempo real (matplotlib)")
    print(f"Cámara: {args.camera}")
    print("Vector de features: 258 dimensiones por frame")
    print("  pose:       132  (33 puntos × x,y,z,vis)")
    print("  left_hand:   63  (21 puntos × x,y,z)")
    print("  right_hand:  63  (21 puntos × x,y,z)")
    print()
    print("Cerrá la ventana para salir.")

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    img_artist = None

    prev_time = time.time()

    with get_holistic_model() as holistic:
        while plt.fignum_exists(fig.number):
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer frame de la cámara.")
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True

            draw_landmarks(image_rgb, results)
            keypoints = extract_keypoints(results)

            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-9)
            prev_time = now

            pose_detected = np.any(keypoints[:132] != 0)
            lh_detected = np.any(keypoints[132:195] != 0)
            rh_detected = np.any(keypoints[195:] != 0)

            title = (
                f"FPS: {fps:.1f} | "
                f"Pose: {'OK' if pose_detected else '--'} | "
                f"Mano Izq: {'OK' if lh_detected else '--'} | "
                f"Mano Der: {'OK' if rh_detected else '--'}"
            )

            if img_artist is None:
                img_artist = ax.imshow(image_rgb)
                ax.axis("off")
            else:
                img_artist.set_data(image_rgb)

            ax.set_title(title, fontsize=10)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

    cap.release()
    plt.close(fig)
    print("Saliendo.")


if __name__ == "__main__":
    main()