"""
Runner de inferencia para señas DINÁMICAS (modelo LSTM, signa_model_vX.tflite).

Hermano de alphabet_runner.py, pero el problema es distinto: una seña dinámica
no está en un frame, está en el movimiento. Por eso acá se mantiene una ventana
de los últimos 2,5 s de landmarks (pose + manos) y se clasifica la secuencia
entera.

Es una copia, paso por paso, de lo que hace la app nativa
(signa-mobile/modules/signa-vision: Detectores.kt, Ventana.kt, Reconocedor.kt
y Confirmador.kt), con el mismo .tflite y los mismos umbrales:

  - MediaPipe Tasks (HandLandmarker + PoseLandmarker) en modo imagen, con la
    pose cada 5 frames: los mismos .task con los que se grabó el dataset.
    Antes este runner usaba Holistic legacy, que es otro modelo: el v9 recibía
    landmarks de un detector que nunca vio.
  - La ventana es de TIEMPO, no de cantidad de frames: los últimos 2,5 s
    remuestreados a 30 pasos. A otros fps la seña llega igual de larga.
  - Los frames sin manos no entran a la ventana (126 ceros donde el modelo
    espera una seña lo hacen responder cualquier cosa con mucha confianza).
  - La pose se usa para normalizar y después se anula: el modelo se entrenó así.
  - Verificación y no identificación: se mira la probabilidad de las señas
    pedidas contra el umbral calibrado de cada una, con un piso de movimiento.
  - Se confirma cuando la seña pasó su umbral en el 60% de los últimos 700 ms.
  - "reposo" ganando libera la confirmación: es la señal de que terminó la seña.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.data.normalization import normalize_keypoints

ROOT = Path(__file__).parent.parent.parent
EXPORTS_DIR = ROOT / "models" / "exports"
MEDIAPIPE_DIR = ROOT.parent / "signa-mobile" / "assets" / "mediapipe"

REST_LABEL = "reposo"

POSE_DIM = 33 * 4
HAND_DIM = 21 * 3
DIM = POSE_DIM + 2 * HAND_DIM

# Los mismos números que Reconocedor.kt / Ventana.kt / Confirmador.kt.
WINDOW_MS = 2500
MIN_SAMPLES = 10
MAX_GAP_MS = 500
MS_BETWEEN_INFERENCES = 80
SMOOTHING = 5
MIN_MOVEMENT = 0.004
CONFIRM_MS = 700
CONFIRM_RATIO = 0.6
POSE_EVERY = 5


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


def load_sign_thresholds(tflite_path: Path) -> dict[str, float]:
    """Umbral por seña de calibrate_signs.py, el mismo que lleva la app en senas.json."""
    path = tflite_path.with_name(tflite_path.stem + "_thresholds.json")
    if not path.exists():
        return {}
    return {k: float(v) for k, v in
            json.loads(path.read_text(encoding="utf-8")).get("thresholds", {}).items()}


class _Window:
    """Ventana.kt: los últimos WINDOW_MS de cuadros con manos, remuestreados por tiempo."""

    def __init__(self, steps: int):
        self.steps = steps
        self.frames: deque[tuple[float, np.ndarray]] = deque()

    def add(self, t: float, kp: np.ndarray | None):
        if kp is not None:
            self.frames.append((t, kp))
        while self.frames and t - self.frames[0][0] > WINDOW_MS:
            self.frames.popleft()

    def clear(self):
        self.frames.clear()

    def usable(self, t: float) -> bool:
        if len(self.frames) < MIN_SAMPLES:
            return False
        if t - self.frames[0][0] < WINDOW_MS * 0.9:
            return False
        prev = self.frames[0][0]
        for ft, _ in self.frames:
            if ft - prev > MAX_GAP_MS:
                return False
            prev = ft
        return True

    def progress(self, t: float) -> float:
        if len(self.frames) < 2:
            return 0.0
        return min(1.0, (t - self.frames[0][0]) / WINDOW_MS, len(self.frames) / MIN_SAMPLES)

    def resample(self, now: float) -> np.ndarray:
        start = now - WINDOW_MS
        frames = list(self.frames)
        out = np.zeros((self.steps, DIM), dtype=np.float32)
        j = 0
        for i in range(self.steps):
            t = start + WINDOW_MS * i / (self.steps - 1)
            while j < len(frames) - 2 and frames[j + 1][0] < t:
                j += 1
            ta, a = frames[j]
            tb, b = frames[min(j + 1, len(frames) - 1)]
            span = tb - ta
            u = min(1.0, max(0.0, (t - ta) / span)) if span > 0 else 0.0
            out[i] = a + (b - a) * u
        return out

    @staticmethod
    def movement(seq: np.ndarray) -> float:
        """Desplazamiento medio de las manos entre pasos."""
        hands = seq[:, POSE_DIM:]
        return float(np.abs(np.diff(hands, axis=0)).mean())


class _Confirmer:
    """Confirmador.kt: la seña pasó su umbral la mayor parte de los últimos 700 ms."""

    def __init__(self):
        self.votes: deque[tuple[float, bool]] = deque()
        self.candidate = None
        self.confirmed = None

    def vote(self, t: float, sign: str | None, ok: bool) -> str | None:
        if sign != self.candidate:
            self.votes.clear()
            self.candidate = sign
        if sign is None:
            return None
        self.votes.append((t, ok))
        while self.votes and t - self.votes[0][0] > CONFIRM_MS:
            self.votes.popleft()
        covers = len(self.votes) > 1 and t - self.votes[0][0] >= CONFIRM_MS * 0.9
        ratio = sum(v for _, v in self.votes) / len(self.votes)
        if covers and ratio >= CONFIRM_RATIO and sign != self.confirmed:
            self.confirmed = sign
            return sign
        return None

    def reset(self):
        self.votes.clear()
        self.candidate = None
        self.confirmed = None


class _HandAssigner:
    """ReconocedorView.repartir: izquierda/derecha por cercanía al cuadro anterior."""

    def __init__(self):
        self.prev_l = None
        self.prev_r = None

    @staticmethod
    def _dist(hand, prev):
        if hand is None or prev is None:
            return float("inf")
        return float(np.hypot(hand[0].x - prev[0], hand[0].y - prev[1]))

    def assign(self, res):
        hands = list(res.hand_landmarks) if res else []
        if not hands:
            self.prev_l = self.prev_r = None
            return None, None
        left = right = None
        if len(hands) >= 2 and self.prev_l and self.prev_r:
            direct = self._dist(hands[0], self.prev_l) + self._dist(hands[1], self.prev_r)
            crossed = self._dist(hands[1], self.prev_l) + self._dist(hands[0], self.prev_r)
            left, right = (hands[0], hands[1]) if direct <= crossed else (hands[1], hands[0])
        elif len(hands) == 1 and (self.prev_l or self.prev_r):
            h = hands[0]
            if self._dist(h, self.prev_l) <= self._dist(h, self.prev_r):
                left = h
            else:
                right = h
        else:
            for i, lms in enumerate(res.hand_landmarks):
                side = res.handedness[i][0].category_name
                if side == "Left":
                    left = lms
                elif side == "Right":
                    right = lms
        self.prev_l = (left[0].x, left[0].y) if left else None
        self.prev_r = (right[0].x, right[0].y) if right else None
        return left, right


class SignRecognizer:
    """
    Reconocedor de señas dinámicas sobre un stream de frames, igual que la app.

    Uso:
        rec = SignRecognizer()
        for frame in camara:                       # BGR, ya espejado
            r = rec.process(frame, targets=["papa", "mama"])
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

    def __init__(self, model_path: Path | None = None, mediapipe_dir: Path | None = None,
                 pose_ignored: bool = True):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        from src.inference.alphabet_runner import get_interpreter

        self.model_path, self.meta = load_latest_sign_model(model_path)
        labels = self.meta.get("labels", {})
        self.labels = [labels[str(i)] for i in range(len(labels))] if labels else []
        self.thresholds = load_sign_thresholds(self.model_path)
        # senas.json · poseIgnored: train.py anula la pose por defecto (drop_pose).
        self.pose_ignored = pose_ignored

        self.interpreter = get_interpreter(self.model_path)
        self.interpreter.allocate_tensors()
        self._in = self.interpreter.get_input_details()[0]
        self._out = self.interpreter.get_output_details()[0]
        self.sequence_length = int(self.meta.get("input_shape", [30, DIM])[0])

        models_dir = Path(mediapipe_dir) if mediapipe_dir else MEDIAPIPE_DIR
        base = lambda name: mp_python.BaseOptions(model_asset_path=str(models_dir / name))
        # Detectores.kt: modo imagen, dos manos, pisos de presencia bajos.
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base("hand_landmarker.task"),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            ))
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base("pose_landmarker.task"),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
            ))

        self._window = _Window(self.sequence_length)
        self._confirmer = _Confirmer()
        self._assigner = _HandAssigner()
        self._smooth: deque = deque(maxlen=SMOOTHING)
        self._frames = 0
        self._pose_res = None
        self._last_inference = 0.0
        self._last = None          # último Result con inferencia, para los frames intermedios

    def close(self):
        self.hands.close()
        self.pose.close()

    def reset(self):
        """Vacía la ventana. Se llama al confirmar una seña para no re-disparar."""
        self._window.clear()
        self._smooth.clear()
        self._confirmer.reset()
        self._last = None

    def track(self, frame_bgr: np.ndarray):
        """
        Sólo pose y manos para dibujar, sin tocar la ventana ni la confirmación.

        Con el reconocimiento en pausa la demo sigue mostrando el esqueleto; lo
        que pasa mientras tanto no entra a la ventana, así que al reanudar no
        se confirma nada con movimiento de antes de la pausa.
        Devuelve (pose_px, hands_px, hay_cuerpo).
        """
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._pose_res is None or self._frames % POSE_EVERY == 0:
            self._pose_res = self.pose.detect(image)
        self._frames += 1
        left, right = self._assigner.assign(self.hands.detect(image))
        pose_px, hands_px = self._landmarks_px(self._pose_res, left, right)
        return pose_px, hands_px, bool(self._pose_res and self._pose_res.pose_landmarks)

    def process(self, frame_bgr: np.ndarray,
                targets: list[str] | None = None) -> "SignRecognizer.Result":
        import mediapipe as mp
        from src.data.tasks_extractor import armar_keypoints

        t = time.monotonic() * 1000
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._pose_res is None or self._frames % POSE_EVERY == 0:
            self._pose_res = self.pose.detect(image)
        self._frames += 1
        hands_res = self.hands.detect(image)
        left, right = self._assigner.assign(hands_res)

        kp = None
        if left is not None or right is not None:
            kp = normalize_keypoints(armar_keypoints(self._pose_res, left, right))
        self._window.add(t, kp)

        pose_px, hands_px = self._landmarks_px(self._pose_res, left, right)
        body_present = bool(self._pose_res and self._pose_res.pose_landmarks)
        n = len(self.labels)

        def result(sign=None, conf=0.0, probs=None, confirmed=False, resting=False, progress=1.0):
            return self.Result(sign, conf, np.zeros(n) if probs is None else probs, confirmed,
                               progress, pose_px, hands_px, body_present, resting)

        if not self._window.usable(t):
            self._last = None
            return result(progress=self._window.progress(t))

        # La app infiere a lo sumo cada 80 ms: la ventana remuestreada casi no
        # cambia entre frames más cercanos, y el suavizado de 5 inferencias y la
        # confirmación de 700 ms están medidos con ese ritmo.
        if t - self._last_inference < MS_BETWEEN_INFERENCES and self._last is not None:
            prev = self._last
            return result(prev.sign, prev.confidence, prev.probs, False, prev.resting)
        self._last_inference = t

        seq = self._window.resample(t)
        x = seq.copy()
        if self.pose_ignored:
            x[:, :POSE_DIM] = 0.0
        self.interpreter.set_tensor(self._in["index"], x[None])
        self.interpreter.invoke()
        self._smooth.append(self.interpreter.get_tensor(self._out["index"])[0])
        smooth = np.mean(self._smooth, axis=0)

        best = self.labels[int(np.argmax(smooth))]
        if best == REST_LABEL:
            self._confirmer.reset()
            self._last = result(best, float(smooth.max()), smooth, resting=True)
            return self._last

        wanted = [s for s in (targets or self.labels) if s in self.labels and s != REST_LABEL]
        cand, p = None, 0.0
        for s in wanted:
            ps = float(smooth[self.labels.index(s)])
            if ps > p:
                cand, p = s, ps

        moving = self._window.movement(seq) >= MIN_MOVEMENT
        ok = cand is not None and p >= self.thresholds.get(cand, 0.5) and moving
        confirmed = self._confirmer.vote(t, cand, ok)
        self._last = result(cand, p, smooth)
        return result(cand, p, smooth, confirmed=confirmed is not None)

    @staticmethod
    def _landmarks_px(pose_res, left, right):
        """Puntos para dibujar, normalizados al frame: pose + las dos manos."""
        pose_px = None
        if pose_res and pose_res.pose_landmarks:
            pose_px = [[round(p.x, 5), round(p.y, 5), round(p.visibility or 0.0, 3)]
                       for p in pose_res.pose_landmarks[0]]
        hands_px = {}
        for name, lms in (("left", left), ("right", right)):
            if lms:
                hands_px[name] = [[round(p.x, 5), round(p.y, 5)] for p in lms]
        return pose_px, hands_px
