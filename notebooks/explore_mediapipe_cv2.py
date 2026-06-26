"""
Exploración interactiva de MediaPipe Holistic en tiempo real (versión cv2.imshow).

Muestra en pantalla:
  - Landmarks de pose (cuerpo completo)
  - Landmarks de manos (izquierda y derecha)
  - Vector de features extraído (dimensiones y preview de valores)
  - FPS en tiempo real

Uso:
    python notebooks/explore_mediapipe_cv2.py
    python notebooks/explore_mediapipe_cv2.py --camera 1   # cámara alternativa

Controles:
    Q     → salir
    S     → guardar screenshot del frame actual en notebooks/screenshots/
    D     → toggle panel de debug (muestra valores raw de landmarks) 
    P     → toggle pose landmarks
    H     → toggle hand landmarks

Notas multiplataforma:
  - Windows: cv2.imshow funciona out-of-the-box, no requiere configuración.
  - Linux + Wayland: si la ventana aparece en negro o tira errores de Qt,
    corré con: QT_QPA_PLATFORM=xcb python notebooks/explore_mediapipe_cv2.py
    Si eso tampoco funciona, usá notebooks/explore_mediapipe.py (versión matplotlib).
"""
import argparse
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.extractor import extract_keypoints, get_holistic_model, draw_landmarks

SCREENSHOT_DIR = Path(__file__).parent / "screenshots"


def maybe_fix_linux_qt_backend():
    """
    En Linux con sesiones Wayland, el plugin Qt 'wayland' embebido en
    opencv-python suele no encontrarse. Forzamos 'xcb' (XWayland) si no
    hay nada seteado, lo cual suele resolverlo sin romper nada en X11 puro.
    No afecta a Windows ni Mac.
    """
    if platform.system() == "Linux" and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def draw_debug_panel(frame, keypoints: np.ndarray, fps: float, show_debug: bool):
    h, w = frame.shape[:2]

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Features: {keypoints.shape[0]}d", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    pose_detected = np.any(keypoints[:132] != 0)
    lh_detected = np.any(keypoints[132:195] != 0)
    rh_detected = np.any(keypoints[195:] != 0)

    status_y = 55
    for label, detected in [("Pose", pose_detected), ("Mano Izq", lh_detected), ("Mano Der", rh_detected)]:
        color = (0, 255, 0) if detected else (0, 0, 200)
        symbol = "OK" if detected else "--"
        cv2.putText(frame, f"{symbol} {label}", (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        status_y += 25

    if show_debug:
        panel_x = w - 200
        cv2.rectangle(frame, (panel_x, 40), (w - 5, 200), (30, 30, 30), -1)
        cv2.putText(frame, "Vector (primeros 15):", (panel_x + 5, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        for i, val in enumerate(keypoints[:15]):
            cv2.putText(frame, f"[{i:02d}] {val:.3f}", (panel_x + 5, 75 + i * 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 220, 180), 1)

    instructions = "[Q] Salir  [S] Screenshot  [D] Debug  [P] Pose  [H] Manos"
    cv2.putText(frame, instructions, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    return frame


def main():
    maybe_fix_linux_qt_backend()

    parser = argparse.ArgumentParser(description="Exploración de MediaPipe Holistic en tiempo real.")
    parser.add_argument("--camera", type=int, default=0, help="Índice de cámara")
    args = parser.parse_args()

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la cámara {args.camera}")
        sys.exit(1)

    print("MediaPipe Holistic — Exploración en tiempo real")
    print(f"Cámara: {args.camera}")
    print("Vector de features: 258 dimensiones por frame")
    print("  pose:       132  (33 puntos × x,y,z,vis)")
    print("  left_hand:   63  (21 puntos × x,y,z)")
    print("  right_hand:  63  (21 puntos × x,y,z)")
    print()

    show_debug = False
    show_pose = True
    show_hands = True
    prev_time = time.time()
    screenshot_count = 0

    with get_holistic_model() as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer frame de la cámara.")
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if show_pose or show_hands:
                draw_landmarks(frame, results)

            keypoints = extract_keypoints(results)

            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-9)
            prev_time = now

            draw_debug_panel(frame, keypoints, fps, show_debug)
            cv2.imshow("Signa - MediaPipe Holistic", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                path = SCREENSHOT_DIR / f"frame_{screenshot_count:03d}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"Screenshot guardado: {path}")
                screenshot_count += 1
            elif key == ord("d"):
                show_debug = not show_debug
            elif key == ord("p"):
                show_pose = not show_pose
            elif key == ord("h"):
                show_hands = not show_hands

    cap.release()
    cv2.destroyAllWindows()
    print("Saliendo.")


if __name__ == "__main__":
    main()
