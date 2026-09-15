"""
Runner de inferencia para señas DINÁMICAS (modelo LSTM, signa_model_vX.tflite).

Hermano de alphabet_runner.py, pero el problema es distinto: una seña dinámica
no está en un frame, está en el movimiento. Por eso acá se mantiene una ventana
deslizante de 30 frames de landmarks Holistic (pose + manos) y se clasifica la
secuencia entera.

Diferencias con el abecedario que importan en vivo:

  - MediaPipe Holistic es bastante más pesado que Hands. Corre a menos fps, y
    como la ventana es de 30 frames fijos, a menos fps la ventana cubre más
    tiempo real. El modelo tolera bastante ese estiramiento porque el dataset
    se aumentó con time_warp, pero conviene no bajar de ~10 fps.

  - Hay una clase "reposo" que es justamente "no estoy haciendo ninguna seña".
    Es la que evita que el modelo dispare cualquier cosa cuando la persona está
    quieta, así que se usa como puerta: mientras gana reposo, no se confirma
    nada.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.data.extractor import extract_keypoints
from src.data.normalization import normalize_keypoints

EXPORTS_DIR = Path(__file__).parent.parent.parent / "models" / "exports"

REST_LABEL = "reposo"


def load_latest_sign_model(explicit: Path | None = None) -> tuple[Path, dict]:
    if explicit is not None:
        path = Path(explicit)
    else:
        candidates = sorted(
            EXPORTS_DIR.glob("signa_model_v*.tflite"),
            key=lambda p: int("".join(c for c in p.stem.split("_v")[-1] if c.isdigit()) or 0),
        )
        if not candidates:
            raise FileNotFoundError(
                "No hay modelos de señas dinámicas en models/exports/. "
                "Corré: python scripts/train.py --model lstm"
            )
        path = candidates[-1]

    meta_path = path.with_name(path.stem + "_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return path, meta


class SignRecognizer:
    """
    Reconocedor de señas dinámicas sobre un stream de frames.

    Uso:
        rec = SignRecognizer()
        for frame in camara:
            r = rec.process(frame)
            r.sign, r.confidence, r.confirmed, r.progress
    """

    class Result:
        __slots__ = ("sign", "confidence", "probs", "confirmed", "progress",
                     "pose_px", "hands_px", "body_present", "resting")

        def __init__(self, sign, confidence, probs, confirmed, progress,
                     pose_px, hands_px, body_present, resting):
            self.sign = sign
            self.confidence = confidence
            self.probs = probs
            self.confirmed = confirmed
            self.progress = progress
            self.pose_px = pose_px
            self.hands_px = hands_px
            self.body_present = body_present
            self.resting = resting

    def __init__(self, model_path: Path | None = None, threshold: float = 0.85,
                 confirm_frames: int = 4, smoothing_window: int = 5):
        import mediapipe as mp

        from src.inference.alphabet_runner import get_interpreter

        self.model_path, self.meta = load_latest_sign_model(model_path)
        labels = self.meta.get("labels", {})
        self.labels = [labels[str(i)] for i in range(len(labels))] if labels else []

        self.interpreter = get_interpreter(self.model_path)
        self.interpreter.allocate_tensors()
        self._in = self.interpreter.get_input_details()[0]
        self._out = self.interpreter.get_output_details()[0]
        self.sequence_length = int(self.meta.get("input_shape", [30, 258])[0])

        self.threshold = threshold
        self.confirm_frames = confirm_frames

        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

        self._window: deque = deque(maxlen=self.sequence_length)
        self._smooth: deque = deque(maxlen=smoothing_window)
        self._streak = 0
        self._last = None

    def close(self):
        self.holistic.close()

    def reset(self):
        """Vacía la ventana. Se llama al confirmar una seña para no re-disparar."""
        self._window.clear()
        self._smooth.clear()
        self._streak = 0
        self._last = None

    def process(self, frame_bgr: np.ndarray) -> "SignRecognizer.Result":
        h, w = frame_bgr.shape[:2]
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)

        keypoints = extract_keypoints(results)
        self._window.append(normalize_keypoints(keypoints))

        pose_px, hands_px = self._landmarks_px(results, w, h)
        body_present = results.pose_landmarks is not None
        progress = len(self._window) / self.sequence_length

        n = len(self.labels)
        if len(self._window) < self.sequence_length:
            return self.Result(None, 0.0, np.zeros(n), False, progress,
                               pose_px, hands_px, body_present, False)

        sequence = np.array(self._window, dtype=np.float32)[np.newaxis]
        self.interpreter.set_tensor(self._in["index"], sequence)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self._out["index"])[0]

        self._smooth.append(probs)
        smooth = np.mean(self._smooth, axis=0)

        idx = int(np.argmax(smooth))
        sign = self.labels[idx] if idx < len(self.labels) else str(idx)
        confidence = float(smooth[idx])

        # "reposo" ganando = la persona está quieta. No es una seña a confirmar,
        # y además es la señal de que terminó la anterior.
        resting = sign == REST_LABEL
        if resting:
            self._streak = 0
            self._last = None
            return self.Result(sign, confidence, smooth, False, progress,
                               pose_px, hands_px, body_present, True)

        if sign == self._last and confidence >= self.threshold:
            self._streak += 1
        else:
            self._streak = 1 if confidence >= self.threshold else 0
            self._last = sign

        confirmed = self._streak >= self.confirm_frames
        return self.Result(sign, confidence, smooth, confirmed, progress,
                           pose_px, hands_px, body_present, False)

    @staticmethod
    def _landmarks_px(results, w, h):
        """Puntos para dibujar: torso/brazos de pose + las dos manos."""
        pose_px = None
        if results.pose_landmarks:
            pose_px = [[round(lm.x, 5), round(lm.y, 5), round(lm.visibility, 3)]
                       for lm in results.pose_landmarks.landmark]

        hands_px = {}
        for name, lms in (("left", results.left_hand_landmarks),
                          ("right", results.right_hand_landmarks)):
            if lms:
                hands_px[name] = [[round(lm.x, 5), round(lm.y, 5)] for lm in lms.landmark]

        return pose_px, hands_px
