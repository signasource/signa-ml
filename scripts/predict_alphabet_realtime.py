"""
Prueba en vivo del modelo de abecedario LSA (ventana de trabajo, no la demo).

Esta es la herramienta para verificar qué letras andan bien y cuáles no antes
de grabar. La escena bonita para el video va aparte.

Uso:
    python scripts/predict_alphabet_realtime.py
    python scripts/predict_alphabet_realtime.py --camera 1 --threshold 0.6
    python scripts/predict_alphabet_realtime.py --model models/exports/signa_alphabet_v2.tflite

Controles:
    Q → salir
    L → mostrar/ocultar landmarks
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.alphabet_runner import AlphabetRecognizer

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "alphabet_config.yaml"

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def maybe_fix_linux_qt_backend():
    if platform.system() == "Linux" and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def draw_hand(frame, pts):
    for a, b in HAND_EDGES:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (200, 200, 200), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(p), 4, (76, 195, 247), -1, cv2.LINE_AA)


def draw_top_k(frame, labels, probs, k=5):
    order = np.argsort(probs)[::-1][:k]
    x0, y0, bar_w, row_h = 12, 70, 180, 26
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 22),
                  (x0 + bar_w + 96, y0 + k * row_h), (30, 25, 22), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, dst=frame)
    cv2.putText(frame, "top-5", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (170, 170, 170), 1, cv2.LINE_AA)

    for i, idx in enumerate(order):
        y = y0 + i * row_h
        conf = float(probs[idx])
        color = (76, 195, 247) if i == 0 else (120, 110, 100)
        cv2.putText(frame, labels[idx], (x0, y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0 + 28, y + 4), (x0 + 28 + bar_w, y + 18), (60, 55, 50), -1)
        cv2.rectangle(frame, (x0 + 28, y + 4), (x0 + 28 + int(bar_w * conf), y + 18), color, -1)
        cv2.putText(frame, f"{conf:.0%}", (x0 + 34 + bar_w, y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    inf = config["inference"]

    p = argparse.ArgumentParser(description="Reconocimiento en vivo del abecedario LSA.")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=inf["threshold"])
    p.add_argument("--confirm-frames", type=int, default=inf["confirm_frames"])
    p.add_argument("--smoothing", type=int, default=inf["smoothing_window"])
    args = p.parse_args()

    maybe_fix_linux_qt_backend()

    rec = AlphabetRecognizer(
        model_path=args.model,
        threshold=args.threshold,
        confirm_frames=args.confirm_frames,
        smoothing_window=args.smoothing,
    )
    print(f"Modelo: {rec.model_path.name}")
    print(f"Letras: {' '.join(rec.labels)}")
    acc = rec.meta.get("accuracy")
    if acc is not None:
        print(f"Accuracy de test reportada: {acc:.1%}")
    print("\nQ para salir · L para landmarks\n")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir la cámara {args.camera}")
        sys.exit(1)

    show_landmarks = True
    prev = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # efecto espejo
            r = rec.process(frame)

            if show_landmarks and r.landmarks_px is not None:
                draw_hand(frame, r.landmarks_px)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-9)
            prev = now
            cv2.putText(frame, f"FPS {fps:.0f}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

            if not r.hand_present:
                cv2.putText(frame, "sin mano en cuadro", (12, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 220), 1, cv2.LINE_AA)
            else:
                draw_top_k(frame, rec.labels, r.probs)
                h, w = frame.shape[:2]
                color = (110, 235, 130) if r.confirmed else (200, 200, 200)
                cv2.putText(frame, r.letter, (w - 130, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 5, cv2.LINE_AA)
                cv2.putText(frame, f"{r.confidence:.0%}", (w - 130, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            cv2.imshow("Signa - Abecedario LSA", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                show_landmarks = not show_landmarks
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rec.close()


if __name__ == "__main__":
    main()
