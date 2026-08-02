"""
Normalización de landmarks extraídos por MediaPipe Holistic.

  MediaPipe devuelve coordenadas (x, y, z) relativas al frame de cámara (0 a 1).
  Esto significa que si te movés en el encuadre, o estás más cerca/lejos,
  los números cambian aunque la seña sea idéntica.

  La normalización hace los landmarks invariantes a:
    - Posición en el encuadre 
    - Distancia a la cámara

Estructura del vector de 258 valores:
  [0:132]   pose: 33 puntos × (x, y, z, visibility)
  [132:195] left_hand: 21 puntos × (x, y, z)
  [195:258] right_hand: 21 puntos × (x, y, z)

Índices relevantes de pose (dentro del bloque de 33 puntos × 4):
  Punto 11 = hombro izquierdo  → índice 11*4 = 44 (x=44, y=45, z=46)
  Punto 12 = hombro derecho    → índice 12*4 = 48 (x=48, y=49, z=50)
"""

import numpy as np

# Índices en el vector de pose (stride de 4: x,y,z,vis)
_LEFT_SHOULDER_IDX = 11 * 4   # x=44, y=45
_RIGHT_SHOULDER_IDX = 12 * 4  # x=48, y=49

POSE_DIM = 33 * 4    # 132
HAND_DIM = 21 * 3    # 63
TOTAL_DIM = POSE_DIM + HAND_DIM * 2  # 258


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normaliza un vector de keypoints de un solo frame (shape: 258,).

    Pasos:
      1. Calcula el centro entre hombros como punto de referencia.
      2. Resta ese centro a TODAS las coordenadas (x,y,z) del vector,
         excluyendo el canal 'visibility' de pose.
      3. Calcula el ancho de hombros como escala de referencia.
      4. Divide todas las coordenadas por ese ancho.

    Si los hombros no se detectaron (zeros), devuelve el vector sin cambios
    para no propagar NaNs al modelo.

    Args:
        keypoints: array de shape (258,)

    Returns:
        array normalizado de shape (258,) — mismo shape, misma estructura.
    """
    assert keypoints.shape == (TOTAL_DIM,), f"Shape esperado (258,), recibido {keypoints.shape}"

    kp = keypoints.copy()

    # Extraer posición de hombros
    lsx, lsy = kp[_LEFT_SHOULDER_IDX], kp[_LEFT_SHOULDER_IDX + 1]
    rsx, rsy = kp[_RIGHT_SHOULDER_IDX], kp[_RIGHT_SHOULDER_IDX + 1]

    # Si los hombros no se detectaron, no normalizar
    if lsx == 0.0 and rsx == 0.0:
        return kp

    # Centro entre hombros (punto de referencia)
    cx = (lsx + rsx) / 2.0
    cy = (lsy + rsy) / 2.0

    # Escala = distancia entre hombros
    shoulder_width = np.sqrt((rsx - lsx) ** 2 + (rsy - lsy) ** 2)
    if shoulder_width < 1e-6:
        return kp

    # --- Normalizar pose (stride 4: x, y, z, visibility) ---
    pose = kp[:POSE_DIM].reshape(33, 4)
    pose[:, 0] = (pose[:, 0] - cx) / shoulder_width   # x
    pose[:, 1] = (pose[:, 1] - cy) / shoulder_width   # y
    pose[:, 2] = pose[:, 2] / shoulder_width           # z (no tiene offset)
    # visibility (pose[:, 3]) no se toca
    kp[:POSE_DIM] = pose.flatten()

    # --- Normalizar manos (stride 3: x, y, z) ---
    for start in [POSE_DIM, POSE_DIM + HAND_DIM]:
        hand = kp[start:start + HAND_DIM].reshape(21, 3)
        if np.any(hand != 0):  # solo si la mano fue detectada
            hand[:, 0] = (hand[:, 0] - cx) / shoulder_width
            hand[:, 1] = (hand[:, 1] - cy) / shoulder_width
            hand[:, 2] = hand[:, 2] / shoulder_width
            kp[start:start + HAND_DIM] = hand.flatten()

    return kp


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Normaliza una secuencia completa de frames.

    Args:
        sequence: array de shape (T, 258) donde T = frames_per_sign

    Returns:
        array normalizado de shape (T, 258)
    """
    assert sequence.ndim == 2 and sequence.shape[1] == TOTAL_DIM, \
        f"Shape esperado (T, 258), recibido {sequence.shape}"
    return np.array([normalize_keypoints(frame) for frame in sequence])


def normalize_dataset(X: np.ndarray) -> np.ndarray:
    """
    Normaliza un dataset completo.

    Args:
        X: array de shape (N, T, 258) donde N = número de secuencias

    Returns:
        array normalizado de shape (N, T, 258)
    """
    assert X.ndim == 3 and X.shape[2] == TOTAL_DIM, \
        f"Shape esperado (N, T, 258), recibido {X.shape}"
    return np.array([normalize_sequence(seq) for seq in X])
