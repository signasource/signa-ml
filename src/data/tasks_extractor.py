"""
Extractor de landmarks con los MISMOS detectores que usa la app.

Por qué no `extractor.py` (Holistic): el dataset viejo se extrajo con
MediaPipe Holistic y la app corre PoseLandmarker + HandLandmarker de MediaPipe
Tasks. Son modelos distintos, y cualquier diferencia entre lo que ve el modelo
al entrenar y lo que ve al reconocer se paga entera en producción. Acá se usan
los mismos .task que viajan dentro de la app, así que los números que salen de
grabar son los mismos que va a ver el teléfono.

El vector es el de siempre, 258 valores por frame:
    pose       33 × (x, y, z, visibility) = 132
    mano izq.  21 × (x, y, z)             =  63
    mano der.  21 × (x, y, z)             =  63
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FEATURE_DIM = POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 * 2


def crear_detectores(models_dir: Path):
    """PoseLandmarker + HandLandmarker en modo VIDEO, como en la app."""
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(models_dir / "pose_landmarker.task")
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )
    manos = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(models_dir / "hand_landmarker.task")
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            # Los mismos pisos que la app: mientras siga viendo la mano, la
            # sigue, en vez de volver a correr el detector de palmas.
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )
    return pose, manos


class AsignadorDeManos:
    """
    Reparte las manos detectadas entre izquierda y derecha.

    La etiqueta de MediaPipe se decide en cada frame por separado y con
    movimiento rápido se equivoca, así que la mano salta de un lado al otro a
    mitad de una seña. Mientras haya historia se asigna por cercanía a dónde
    estaba cada mano un frame antes. Es la misma lógica que la app, a propósito.
    """

    def __init__(self) -> None:
        self.previa_izq = None
        self.previa_der = None

    @staticmethod
    def _dist(mano, previa) -> float:
        if mano is None or previa is None:
            return float("inf")
        return float(np.hypot(mano[0].x - previa[0], mano[0].y - previa[1]))

    def _por_etiqueta(self, res):
        izq = der = None
        for i, lms in enumerate(res.hand_landmarks):
            lado = res.handedness[i][0].category_name
            if lado == "Left":
                izq = lms
            elif lado == "Right":
                der = lms
        return izq, der

    def asignar(self, res):
        manos = list(res.hand_landmarks) if res else []
        if not manos:
            self.previa_izq = self.previa_der = None
            return None, None

        if len(manos) >= 2 and self.previa_izq and self.previa_der:
            directo = self._dist(manos[0], self.previa_izq) + self._dist(manos[1], self.previa_der)
            cruzado = self._dist(manos[1], self.previa_izq) + self._dist(manos[0], self.previa_der)
            izq, der = (manos[0], manos[1]) if directo <= cruzado else (manos[1], manos[0])
        elif len(manos) == 1 and (self.previa_izq or self.previa_der):
            m = manos[0]
            if self._dist(m, self.previa_izq) <= self._dist(m, self.previa_der):
                izq, der = m, None
            else:
                izq, der = None, m
        else:
            izq, der = self._por_etiqueta(res)

        self.previa_izq = (izq[0].x, izq[0].y) if izq else None
        self.previa_der = (der[0].x, der[0].y) if der else None
        return izq, der


def armar_keypoints(pose_res, izq, der) -> np.ndarray:
    """Los 258 valores del frame, con ceros donde no se detectó nada."""
    if pose_res and pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks[0]
        pose = np.array([[p.x, p.y, p.z, p.visibility] for p in lms]).flatten()
    else:
        pose = np.zeros(POSE_LANDMARKS * 4)

    def mano(lms):
        if not lms:
            return np.zeros(HAND_LANDMARKS * 3)
        return np.array([[p.x, p.y, p.z] for p in lms]).flatten()

    return np.concatenate([pose, mano(izq), mano(der)]).astype(np.float32)


# ─── Modo imagen: una foto suelta, para el abecedario ────────────────────────

# Índices de BlazePose que usamos como referencia de cara. La app ya corre el
# detector de pose, así que sacar de ahí los ojos y la boca evita meterle un
# tercer modelo sólo para ubicar la mano respecto de la cara.
OJO_IZQ, OJO_DER = 2, 5
BOCA_IZQ, BOCA_DER = 9, 10


def detectores_estaticos(models_dir: Path):
    """Los mismos dos modelos, en modo IMAGE, para fotos sueltas."""
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(models_dir / "pose_landmarker.task")
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
    )
    manos = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(models_dir / "hand_landmarker.task")
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.4,
        )
    )
    return pose, manos


def mano_principal(res):
    """(landmarks 21x3, world 21x3, 'Left'|'Right', score) de la mano detectada."""
    if not res or not res.hand_landmarks:
        return None
    lm = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]])
    world = None
    if getattr(res, "hand_world_landmarks", None):
        world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]])
    cat = res.handedness[0][0]
    return lm, world, cat.category_name, float(cat.score)


def referencia_cara(pose_res):
    """
    (centro de los ojos, centro de la boca) sacados de la pose.

    El dataset viejo los tomaba de BlazeFace, un detector más que la app no
    lleva. BlazePose ya trae ojos y boca entre sus 33 puntos, así que salen de
    ahí: son puntos distintos de los de BlazeFace, y por eso hay que reextraer
    y reentrenar — mezclar las dos fuentes sería el mismo desajuste de siempre.
    """
    if not pose_res or not pose_res.pose_landmarks:
        return None, None
    p = pose_res.pose_landmarks[0]
    ojo = np.array([(p[OJO_IZQ].x + p[OJO_DER].x) / 2,
                    (p[OJO_IZQ].y + p[OJO_DER].y) / 2])
    boca = np.array([(p[BOCA_IZQ].x + p[BOCA_DER].x) / 2,
                     (p[BOCA_IZQ].y + p[BOCA_DER].y) / 2])
    return ojo, boca
