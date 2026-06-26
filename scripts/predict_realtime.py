"""
Inferencia en tiempo real usando el modelo .tflite exportado.

Versión optimizada para FPS altos: usa cv2.imshow + overlay dibujado
directamente sobre el frame (en vez de un panel matplotlib separado),
que es mucho más rápido para video en vivo.

Usa una ventana deslizante de N frames para predecir señas continuamente.
La detección se confirma cuando la confianza supera un umbral durante
varios frames consecutivos.

Uso:
    python scripts/predict_realtime.py
    python scripts/predict_realtime.py --model models/exports/signa_model_v1.tflite
    python scripts/predict_realtime.py --camera 1 --threshold 0.85

Controles:
    Q → salir
"""
import argparse
import json
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.extractor import get_holistic_model, extract_keypoints, draw_landmarks
from src.data.normalization import normalize_keypoints

EXPORTS_DIR = Path(__file__).parent.parent / "models" / "exports"

# Paleta simple para las barras de confianza
BAR_COLOR = (244, 138, 76)        # naranja BGR
BAR_COLOR_TOP = (76, 195, 247)    # celeste BGR (clase con mayor confianza)
BG_PANEL = (35, 25, 20)           # fondo oscuro del panel


def maybe_fix_linux_qt_backend():
    if platform.system() == "Linux" and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def load_latest_model() -> tuple[Path, dict]:
    """Carga el .tflite más reciente y su metadata."""
    tflite_files = sorted(EXPORTS_DIR.glob("signa_model_v*.tflite"))
    if not tflite_files:
        print("ERROR: No hay modelos exportados en models/exports/")
        print("Corré primero: python scripts/train.py")
        sys.exit(1)

    tflite_path = tflite_files[-1]
    meta_path = tflite_path.with_name(tflite_path.stem + "_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"labels": {}}
    return tflite_path, meta


def get_interpreter(tflite_path: Path):
    """Intenta tflite_runtime (liviano) y cae a tensorflow.lite si no está."""
    try:
        import tflite_runtime.interpreter as tflite  # type: ignore
        return tflite.Interpreter(model_path=str(tflite_path))
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=str(tflite_path))


def draw_confidence_panel(frame: np.ndarray, labels: list[str], confidences: np.ndarray, threshold: float):
    """
    Dibuja un panel de barras de confianza directamente sobre el frame con cv2.
    Mucho más rápido que un gráfico matplotlib separado.
    """
    h, w = frame.shape[:2]
    panel_w = 220
    panel_x0 = w - panel_w - 10
    panel_y0 = 10
    bar_h = 22
    gap = 8
    panel_h = len(labels) * (bar_h + gap) + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x0, panel_y0), (w - 10, panel_y0 + panel_h), BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, dst=frame)

    top_idx = int(np.argmax(confidences)) if len(confidences) else -1

    for i, (label, conf) in enumerate(zip(labels, confidences)):
        y = panel_y0 + 8 + i * (bar_h + gap)
        bar_max_w = panel_w - 70
        bar_w = int(bar_max_w * conf)
        color = BAR_COLOR_TOP if i == top_idx else BAR_COLOR

        cv2.rectangle(frame, (panel_x0 + 65, y), (panel_x0 + 65 + bar_max_w, y + bar_h - 4),
                      (60, 60, 60), -1)
        cv2.rectangle(frame, (panel_x0 + 65, y), (panel_x0 + 65 + bar_w, y + bar_h - 4),
                      color, -1)
        cv2.putText(frame, label[:8], (panel_x0 + 4, y + bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(frame, f"{conf:.0%}", (panel_x0 + 65 + bar_max_w + 4, y + bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Línea de umbral
    thresh_x = panel_x0 + 65 + int((panel_w - 70) * threshold)
    cv2.line(frame, (thresh_x, panel_y0 + 5), (thresh_x, panel_y0 + panel_h - 5), (50, 220, 255), 1)


def run_inference(tflite_path, label_map, camera_index, sequence_length, threshold, confirm_frames):
    maybe_fix_linux_qt_backend()

    interpreter = get_interpreter(tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    labels = [label_map.get(str(i), label_map.get(i, f"clase_{i}"))
              for i in range(len(label_map))]

    window: deque = deque(maxlen=sequence_length)
    last_prediction = None
    confirm_counter = 0
    confirmed_sign = None
    confirmed_at = 0.0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la cámara {camera_index}")
        sys.exit(1)

    print(f"\nModelo: {tflite_path.name}")
    print(f"Señas: {labels}")
    print(f"Umbral: {threshold:.0%} | Ventana: {sequence_length} frames\n")
    print("Presioná Q para salir.")

    prev_time = time.time()
    confidences = np.zeros(len(labels))

    with get_holistic_model() as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            draw_landmarks(frame, results)

            keypoints = extract_keypoints(results)
            keypoints = normalize_keypoints(keypoints)
            window.append(keypoints)

            prediction_label = None

            if len(window) == sequence_length:
                sequence = np.array(window, dtype=np.float32)[np.newaxis]  # (1, T, 258)
                interpreter.set_tensor(input_details[0]["index"], sequence)
                interpreter.invoke()
                confidences = interpreter.get_tensor(output_details[0]["index"])[0]

                pred_idx = int(np.argmax(confidences))
                pred_conf = float(confidences[pred_idx])
                prediction_label = labels[pred_idx]

                if prediction_label == last_prediction and pred_conf >= threshold:
                    confirm_counter += 1
                else:
                    confirm_counter = 1
                    last_prediction = prediction_label

                if confirm_counter >= confirm_frames and pred_conf >= threshold:
                    confirmed_sign = prediction_label
                    confirmed_at = time.time()
                    confirm_counter = 0
                    print(f"✓ Seña detectada: {confirmed_sign}  ({pred_conf:.1%})")

            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-9)
            prev_time = now

            # --- Overlay ---
            h, w = frame.shape[:2]
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(frame, f"Buffer: {len(window)}/{sequence_length}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if len(window) == sequence_length:
                draw_confidence_panel(frame, labels, confidences, threshold)

            if confirmed_sign and (time.time() - confirmed_at < 2.0):
                cv2.putText(frame, confirmed_sign.upper(), (w // 2 - 100, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (50, 220, 255), 3)

            cv2.imshow("Signa - Reconocimiento en tiempo real", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Saliendo.")


def main():
    parser = argparse.ArgumentParser(description="Inferencia en tiempo real con el modelo .tflite.")
    parser.add_argument("--model", type=str, default=None,
                        help="Ruta al .tflite (default: el más reciente en models/exports/)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--confirm-frames", type=int, default=3)
    args = parser.parse_args()

    if args.model:
        tflite_path = Path(args.model)
        meta_path = tflite_path.with_name(tflite_path.stem + "_meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    else:
        tflite_path, meta = load_latest_model()

    label_map = meta.get("labels", {})
    sequence_length = meta.get("input_shape", [30, 258])[0]

    if not label_map:
        print("ADVERTENCIA: No se encontró metadata con labels.")

    run_inference(
        tflite_path=tflite_path,
        label_map=label_map,
        camera_index=args.camera,
        sequence_length=sequence_length,
        threshold=args.threshold,
        confirm_frames=args.confirm_frames,
    )


if __name__ == "__main__":
    main()