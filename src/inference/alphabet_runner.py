"""
Runner de inferencia del modelo de abecedario, compartido por el script de
prueba en vivo y por las escenas de la demo.

Encapsula lo que hay que hacer bien para que el reconocimiento no titile:
    - una sola instancia de MediaPipe Hands en modo tracking (no static)
    - promedio móvil de probabilidades (no del argmax): suaviza sin agregar lag
    - confirmación por N frames consecutivos antes de dar una letra por buena
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.data.hand_features import (build_features, build_face_features, mirror,
                                    FEATURE_DIM, FACE_BLOCK_DIM,
                                    FACE_INDEX_Y, FACE_PRESENT)

EXPORTS_DIR = Path(__file__).parent.parent.parent / "models" / "exports"


def load_thresholds(tflite_path: Path, labels: list[str]) -> dict[str, float]:
    """
    Umbral de aceptación por letra para el modo verificación.

    Los genera scripts/calibrate_alphabet.py y quedan al lado del .tflite. Si no
    están, se cae a 0.5 para todas — anda, pero peor: las letras "tímidas"
    (las que reparten probabilidad con una vecina parecida) casi nunca llegan a
    un umbral único, y las muy seguras aceptan de más.
    """
    path = tflite_path.with_name(tflite_path.stem + "_thresholds.json")
    if not path.exists():
        return {l: 0.5 for l in labels}
    data = json.loads(path.read_text(encoding="utf-8")).get("thresholds", {})
    return {l: float(data.get(l, 0.5)) for l in labels}


def load_latest_alphabet_model(explicit: Path | None = None) -> tuple[Path, dict]:
    """Devuelve (ruta al .tflite, metadata). Por defecto, la versión más alta."""
    if explicit is not None:
        path = Path(explicit)
    else:
        candidates = sorted(
            EXPORTS_DIR.glob("signa_alphabet_v*.tflite"),
            key=lambda p: int("".join(c for c in p.stem.split("_v")[-1] if c.isdigit()) or 0),
        )
        if not candidates:
            raise FileNotFoundError(
                "No hay modelos de abecedario en models/exports/. "
                "Corré: python scripts/train_alphabet.py"
            )
        path = candidates[-1]

    meta_path = path.with_name(path.stem + "_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return path, meta


def get_interpreter(tflite_path: Path):
    """tflite_runtime si está (arranca más rápido), si no tensorflow.lite."""
    try:
        import tflite_runtime.interpreter as tflite  # type: ignore
        return tflite.Interpreter(model_path=str(tflite_path))
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=str(tflite_path))


# Pares de letras que comparten la forma de la mano y sólo se distinguen por
# DÓNDE se apoya: la T en la pera, la I en el ojo. El modelo ya recibe esa
# posición, pero no termina de decidirse — se lo midió prediciendo I con 58%
# sobre una mano que estaba claramente en la pera. Entre esas dos la decisión
# la toma directamente la altura, que en el dataset separa las dos nubes al 99%
# (la I nunca baja de +0.57, la T nunca sube por encima de +0.89).
#
#   (letra de abajo, letra de arriba, umbral)   ojos = 0.0 · boca = 1.0
LOCATION_PAIRS = (("I", "T", 0.73),)


class AlphabetRecognizer:
    """
    Reconocedor de letras estáticas sobre frames de cámara.

    Uso:
        rec = AlphabetRecognizer()
        for frame in camara:
            r = rec.process(frame)   # BGR, ya espejado si querés efecto espejo
            r.letter, r.confidence, r.confirmed, r.landmarks_px
    """

    class Result:
        __slots__ = ("letter", "confidence", "probs", "confirmed",
                     "landmarks_px", "hand_present", "face_present",
                     "target", "target_confidence", "target_ok", "mismatch")

        def __init__(self, letter, confidence, probs, confirmed, landmarks_px, hand_present,
                     target=None, target_confidence=0.0, target_ok=False, mismatch=False,
                     face_present=None):
            self.letter = letter
            self.confidence = confidence
            self.probs = probs
            self.confirmed = confirmed
            self.landmarks_px = landmarks_px
            self.hand_present = hand_present
            # None = al modelo no le hace falta la cara. False = la necesita y
            # no la vio, así que las letras que se distinguen por dónde va la
            # mano (T/I) van a volver a confundirse.
            self.face_present = face_present
            self.target = target                      # letra pedida, si hay
            self.target_confidence = target_confidence
            self.target_ok = target_ok                # p(target) pasó su umbral
            self.mismatch = mismatch                  # sostenidamente hace otra letra

    def __init__(self, model_path: Path | None = None, threshold: float = 0.75,
                 confirm_frames: int = 5, smoothing_window: int = 7,
                 min_detection_confidence: float = 0.6):
        import mediapipe as mp

        self.model_path, self.meta = load_latest_alphabet_model(model_path)
        labels = self.meta.get("labels", {})
        self.labels = [labels[str(i)] for i in range(len(labels))] if labels else []

        self.interpreter = get_interpreter(self.model_path)
        self.interpreter.allocate_tensors()
        self._in = self.interpreter.get_input_details()[0]
        self._out = self.interpreter.get_output_details()[0]

        # Cuántas veces espera el modelo el bloque de posición respecto de la
        # cara. Se deduce de su propia entrada en vez de fijarlo acá, así los
        # modelos viejos (258, sólo mano) siguen andando sin cambios y no hace
        # falta pagar el detector de cara si el modelo no lo pide.
        entrada = int(self._in["shape"][1])
        extra = entrada - FEATURE_DIM
        if extra < 0 or extra % FACE_BLOCK_DIM:
            raise ValueError(
                f"{self.model_path.name} espera {entrada} features y no encaja con "
                f"{FEATURE_DIM} de mano + N×{FACE_BLOCK_DIM} de cara.")
        self.face_repeat = extra // FACE_BLOCK_DIM

        self.face = None
        if self.face_repeat:
            self.face = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.4)
            self._face_kp = mp.solutions.face_detection.FaceKeyPoint

        self.threshold = threshold
        self.confirm_frames = confirm_frames
        self.thresholds = load_thresholds(self.model_path, self.labels)

        # static_image_mode=False activa el tracking entre frames: mucho más
        # rápido y mucho más estable que re-detectar la mano en cada frame.
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

        self._window: deque = deque(maxlen=smoothing_window)
        self._streak = 0
        self._last = None
        self._target_streak = 0
        self._mismatch_streak = 0

    def close(self):
        self.hands.close()
        if self.face is not None:
            self.face.close()

    def _apply_location_rules(self, probs, cara):
        """
        Reparte la probabilidad de cada par entre sus dos letras según la altura.

        Se hace sobre las probabilidades y no sobre el argmax final para que
        valga igual en identificación y en verificación: si la mano está en la
        pera, p(I) queda en cero y la demo deja de aceptar una T como I.
        Sin cara detectada no se toca nada: no hay altura confiable que usar.
        """
        if cara is None or cara[FACE_PRESENT] != 1.0:
            return probs
        altura = float(cara[FACE_INDEX_Y])
        for baja, alta, umbral in LOCATION_PAIRS:
            if baja not in self.labels or alta not in self.labels:
                continue
            i_baja, i_alta = self.labels.index(baja), self.labels.index(alta)
            total = probs[i_baja] + probs[i_alta]
            if total <= 0:
                continue
            gana = i_alta if altura > umbral else i_baja
            pierde = i_baja if gana == i_alta else i_alta
            probs = probs.copy()
            probs[gana], probs[pierde] = total, 0.0
        return probs

    def _face_reference(self, rgb):
        """(centro de ojos, centro de boca) o (None, None) si no se ve la cara."""
        if self.face is None:
            return None, None
        det = self.face.process(rgb)
        if not det.detections:
            return None, None
        kp = det.detections[0].location_data.relative_keypoints
        K = self._face_kp
        ojo = np.array([(kp[K.RIGHT_EYE].x + kp[K.LEFT_EYE].x) / 2,
                        (kp[K.RIGHT_EYE].y + kp[K.LEFT_EYE].y) / 2])
        boca = np.array([kp[K.MOUTH_CENTER].x, kp[K.MOUTH_CENTER].y])
        return ojo, boca

    def reset(self):
        """Olvida el historial — útil al cambiar de letra objetivo en la demo."""
        self._window.clear()
        self._streak = 0
        self._last = None
        self._target_streak = 0
        self._mismatch_streak = 0

    def process(self, frame_bgr: np.ndarray,
                target: str | None = None) -> "AlphabetRecognizer.Result":
        """
        Clasifica un frame.

        Sin `target` funciona en modo IDENTIFICACIÓN: gana la letra más probable
        de las 26. Es lo que necesita una pantalla de "¿qué seña estoy haciendo?".

        Con `target` funciona en modo VERIFICACIÓN: sólo hay que decidir si la
        mano es esa letra, usando su umbral calibrado. Es bastante más fácil que
        elegir entre 26, y es lo que hace que todas las letras sean usables en
        el deletreo — incluso las que en identificación pierden contra una
        vecina parecida.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            self._window.clear()
            self._streak = 0
            self._last = None
            self._target_streak = 0
            self._mismatch_streak = 0
            n = len(self.labels)
            return self.Result(None, 0.0, np.zeros(n), False, None, False, target=target)

        hand = results.multi_hand_landmarks[0]
        lm = np.array([[p.x, p.y, p.z] for p in hand.landmark])
        landmarks_px = np.column_stack([lm[:, 0] * w, lm[:, 1] * h]).astype(np.int32)

        world = None
        if getattr(results, "multi_hand_world_landmarks", None):
            world = np.array([[p.x, p.y, p.z]
                              for p in results.multi_hand_world_landmarks[0].landmark])

        label = "Right"
        if results.multi_handedness:
            label = results.multi_handedness[0].classification[0].label
        espejada = label.lower().startswith("l")

        # Igual que al construir el dataset: la posición se mide con los
        # landmarks SIN espejar y se le avisa si la mano se va a espejar.
        cara = None
        face_present = None
        if self.face_repeat:
            ojo, boca = self._face_reference(rgb)
            face_present = ojo is not None
            cara = build_face_features(lm, ojo, boca, mirrored=espejada)

        if espejada:
            lm = mirror(lm)
            if world is not None:
                world = mirror(world)

        vec = build_features(lm, world)
        if cara is not None:
            vec = np.concatenate([vec] + [cara] * self.face_repeat)
        features = vec[None].astype(np.float32)
        self.interpreter.set_tensor(self._in["index"], features)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self._out["index"])[0]

        probs = self._apply_location_rules(probs, cara)

        # Promediamos las PROBABILIDADES, no el argmax: si dos letras compiten,
        # el promedio decide con la evidencia acumulada en vez de saltar.
        self._window.append(probs)
        smooth = np.mean(self._window, axis=0)

        idx = int(np.argmax(smooth))
        letter = self.labels[idx] if idx < len(self.labels) else str(idx)
        confidence = float(smooth[idx])

        if letter == self._last and confidence >= self.threshold:
            self._streak += 1
        else:
            self._streak = 1 if confidence >= self.threshold else 0
            self._last = letter

        # ─── Modo identificación ────────────────────────────────────────────
        if target is None:
            confirmed = self._streak >= self.confirm_frames
            return self.Result(letter, confidence, smooth, confirmed, landmarks_px, True,
                               face_present=face_present)

        # ─── Modo verificación ──────────────────────────────────────────────
        target = target.upper()
        if target not in self.labels:
            return self.Result(letter, confidence, smooth, False, landmarks_px, True,
                               target=target, face_present=face_present)

        t_idx = self.labels.index(target)
        t_conf = float(smooth[t_idx])
        t_ok = t_conf >= self.thresholds.get(target, 0.5)

        self._target_streak = self._target_streak + 1 if t_ok else 0

        # "Se equivocó" sólo si sostiene con confianza OTRA letra. Un pico
        # suelto mientras acomoda la mano no cuenta: acusar de error a alguien
        # que todavía se está posicionando arruina la práctica.
        other = (letter != target
                 and confidence >= max(self.threshold, self.thresholds.get(letter, 0.5))
                 and not t_ok)
        self._mismatch_streak = self._mismatch_streak + 1 if other else 0

        return self.Result(
            letter, confidence, smooth,
            self._target_streak >= self.confirm_frames,
            landmarks_px, True,
            target=target, target_confidence=t_conf, target_ok=t_ok,
            mismatch=self._mismatch_streak >= self.confirm_frames * 2,
            face_present=face_present,
        )
