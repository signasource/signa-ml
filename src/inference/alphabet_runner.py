"""
Runner de inferencia del modelo de abecedario, compartido por el script de
prueba en vivo y por las escenas de la demo.

Encapsula lo que hay que hacer bien para que el reconocimiento no titile:
    - los MISMOS detectores con los que se armó el dataset y que corre la app
      (MediaPipe Tasks: HandLandmarker + PoseLandmarker, en modo imagen)
    - promedio móvil de probabilidades (no del argmax): suaviza sin agregar lag
    - confirmación por N frames consecutivos antes de dar una letra por buena

Sobre los detectores: este runner usaba las soluciones legacy (`mp.solutions`
Hands + BlazeFace) cuando el dataset ya se extraía con Tasks y la cara salía
de la pose. Son modelos distintos, y el bloque de cara —la mitad de la entrada
del modelo— se calculaba con puntos que el modelo nunca vio al entrenar. Además
el .tflite que levantaba la demo (v3) se había entrenado con un dataset viejo,
extraído con los legacy, mientras que el .keras que va a la app y los umbrales
salían del dataset de Tasks: la demo combinaba un modelo con los umbrales de
otro. Pasado por estos detectores, el v3 acierta 27% de las fotos de su propio
dataset; el .keras, 94%.

Por eso el modelo trae en su metadata con qué detectores se extrajo su dataset
(`detectors`), y acá se avisa si no son estos.
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

ROOT = Path(__file__).parent.parent.parent
EXPORTS_DIR = ROOT / "models" / "exports"
# Los .task viajan con la app; se usan esos mismos para que lo que ve la demo
# sea lo que va a ver el teléfono.
# Los .task viajan con la app; se usan esos mismos para que lo que ve la demo
# sea lo que va a ver el teléfono.
MEDIAPIPE_DIR = ROOT.parent / "signa-mobile" / "assets" / "mediapipe"
DETECTORS = "mediapipe-tasks"


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
# DÓNDE se apoya: la T en la boca, la I al costado del ojo. El modelo ya recibe
# esa posición, pero no termina de decidirse: fuera de muestra puntúa fotos de I
# con p(T) de 0.99. Entre esas dos la decisión la toma la altura de la punta del
# índice, que en el dataset separa las dos nubes (la I nunca pasa de 0.59, la T
# nunca baja de 0.75; el corte va en el medio).
#
# El corte es suave: a LOCATION_SOFTNESS del límite la probabilidad se reparte
# en vez de saltar de una letra a la otra. Con el corte duro, una T con la punta
# del dedo tocando la nariz quedaba con p(T) = 0 y no había forma de que pasara.
#
#   (letra de arriba en la cara, letra de abajo, umbral)   ojos = 0.0 · boca = 1.0
LOCATION_PAIRS = (("I", "T", 0.67),)
LOCATION_SOFTNESS = 0.04


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
                 mediapipe_dir: Path | None = None, pose_every: int = 5):
        from src.data.tasks_extractor import detectores_estaticos

        self.model_path, self.meta = load_latest_alphabet_model(model_path)
        labels = self.meta.get("labels", {})
        self.labels = [labels[str(i)] for i in range(len(labels))] if labels else []

        if self.meta.get("detectors") != DETECTORS:
            print(f"  ⚠ {self.model_path.name} no dice que se haya entrenado con {DETECTORS}: "
                  "si su dataset salió de los detectores legacy, acá va a acertar muy "
                  "poco. Reentrenalo con scripts/train_alphabet.py --with-face.")

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

        self.threshold = threshold
        self.confirm_frames = confirm_frames
        self.thresholds = load_thresholds(self.model_path, self.labels)

        # Los mismos dos detectores, con la misma configuración, que usa
        # build_alphabet_dataset.py: así el modelo ve en vivo exactamente los
        # números con los que se entrenó. Modo imagen, como la app.
        models_dir = Path(mediapipe_dir) if mediapipe_dir else MEDIAPIPE_DIR
        faltan = [n for n in ("hand_landmarker.task", "pose_landmarker.task")
                  if not (models_dir / n).is_file()]
        if faltan:
            raise FileNotFoundError(
                f"Faltan {', '.join(faltan)} en {models_dir}. Son los de la app "
                "(signa-mobile/assets/mediapipe); pasá la carpeta con mediapipe_dir.")
        self.pose, self.hands = detectores_estaticos(models_dir)

        # La pose se refresca 1 de cada `pose_every` frames, igual que en la
        # app: de ella sólo sale la referencia de la cara, y la cabeza se mueve
        # mucho más despacio que la mano. Si el modelo no usa la cara, ni se corre.
        self.pose_every = max(1, pose_every)
        self._frames = 0
        self._pose_res = None

        self._window: deque = deque(maxlen=smoothing_window)
        self._streak = 0
        self._last = None
        self._target_streak = 0
        self._mismatch_streak = 0

    def close(self):
        self.hands.close()
        self.pose.close()

    def _apply_location_rules(self, probs, cara):
        """
        Reparte la probabilidad de cada par entre sus dos letras según la altura.

        Se hace sobre las probabilidades y no sobre el argmax final para que
        valga igual en identificación y en verificación: si la mano está en la
        boca, p(I) queda casi en cero y la demo deja de aceptar una T como I.
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
            # Peso de la letra que va más abajo en la cara (la T): 0.5 justo en
            # el umbral, y casi 1 apenas la punta del dedo baja de ahí.
            peso_alta = 1.0 / (1.0 + np.exp(-(altura - umbral) / LOCATION_SOFTNESS))
            probs = probs.copy()
            probs[i_alta], probs[i_baja] = total * peso_alta, total * (1.0 - peso_alta)
        return probs

    def _face_reference(self, imagen):
        """(centro de ojos, centro de boca) sacados de la pose, o (None, None)."""
        from src.data.tasks_extractor import referencia_cara

        if self._pose_res is None or self._frames % self.pose_every == 0:
            self._pose_res = self.pose.detect(imagen)
        return referencia_cara(self._pose_res)

    def track(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """
        Sólo la mano, sin clasificar: landmarks en píxeles, o None.

        Es lo que usa la demo con el reconocimiento en pausa, para que el
        esqueleto se siga dibujando. No toca el suavizado ni las rachas: al
        reanudar se arranca igual que si nunca se hubiera pausado.
        """
        import mediapipe as mp
        from src.data.tasks_extractor import mano_principal

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        found = mano_principal(self.hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)))
        if found is None:
            return None
        lm = found[0]
        return np.column_stack([lm[:, 0] * w, lm[:, 1] * h]).astype(np.int32)

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
        import mediapipe as mp
        from src.data.tasks_extractor import mano_principal

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        found = mano_principal(self.hands.detect(imagen))

        # La pose se mira aunque no haya mano: así, cuando la mano aparece, la
        # referencia de la cara ya está y el primer frame no sale sin posición.
        ojo = boca = None
        if self.face_repeat:
            ojo, boca = self._face_reference(imagen)
        self._frames += 1

        if found is None:
            self._window.clear()
            self._streak = 0
            self._last = None
            self._target_streak = 0
            self._mismatch_streak = 0
            n = len(self.labels)
            return self.Result(None, 0.0, np.zeros(n), False, None, False, target=target)

        lm, world, label, _score = found
        landmarks_px = np.column_stack([lm[:, 0] * w, lm[:, 1] * h]).astype(np.int32)
        espejada = label.lower().startswith("l")

        # Igual que al construir el dataset: la posición se mide con los
        # landmarks SIN espejar y se le avisa si la mano se va a espejar.
        cara = None
        face_present = None
        if self.face_repeat:
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
