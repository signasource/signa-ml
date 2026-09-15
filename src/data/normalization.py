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


# ─── La pose, fuera de la entrada del modelo ────────────────────────────────

POSE_BLOCK = 132


def drop_pose(X: np.ndarray) -> np.ndarray:
    """
    Pone en cero el bloque de pose, dejando sólo las manos.

    Medido sobre el modelo v6: con las manos en cero acertaba el 100% de papá y
    de hermano, y con la pose en cero caía a 0%. O sea que no clasificaba por la
    seña sino por la postura de brazos y torso — un atajo perfecto dentro de un
    dataset de una sola persona en una sola sesión, e inútil en el teléfono,
    donde la postura es otra. Peor: con las manos abajo el cuerpo igual "parece"
    algo y el modelo disparaba con confianza.

    La pose se sigue usando ANTES, para normalizar: los hombros son el origen y
    la escala, así que las manos ya quedan expresadas respecto del cuerpo. Lo
    que se quita es la postura como evidencia directa.
    """
    out = X.copy()
    out[..., :POSE_BLOCK] = 0.0
    return out


MANOS_DESDE = POSE_BLOCK
MIN_FRAMES_CON_MANOS = 8


def solo_con_manos(seq: np.ndarray) -> np.ndarray | None:
    """
    Se queda con los frames donde MediaPipe vio alguna mano y los reestira a
    los mismos pasos que tenía la secuencia.

    La app dejó de meter cuadros sin manos a la ventana —son 126 ceros donde el
    modelo espera una seña— así que los clips de entrenamiento tampoco pueden
    tenerlos: si no, el modelo aprende el patrón de ausencia en vez de la seña.
    Y era medible: en los clips de "mama" hay manos en apenas el 40% de los
    frames, contra el 74% de "papa".

    Devuelve None si quedan muy pocos frames útiles: ese clip no describe nada.
    """
    con_manos = np.abs(seq[:, MANOS_DESDE:]).sum(axis=1) > 0
    utiles = seq[con_manos]
    if len(utiles) < MIN_FRAMES_CON_MANOS:
        return None
    if len(utiles) == len(seq):
        return seq

    origen = np.linspace(0, 1, len(utiles))
    destino = np.linspace(0, 1, len(seq))
    return np.stack(
        [np.interp(destino, origen, utiles[:, c]) for c in range(seq.shape[1])],
        axis=1,
    ).astype(seq.dtype)


# ─── Encuadre dentro de la ventana ───────────────────────────────────────────

# Cuánto se corre la seña dentro de la ventana, en pasos de la secuencia.
#
# Los clips se grabaron encuadrados: los 30 frames son exactamente la seña. La
# app no tiene ese lujo — su ventana son SIEMPRE los últimos 2,5 s de video, así
# que la seña entra y sale de a poco, con reposo adelante o atrás. Medido sobre
# el modelo v8, corriendo la seña 10 pasos (0,83 s) la probabilidad de 'papa'
# cae de 0.90 a 0.47 y la de 'mama' de 0.80 a 0.28: la mitad de las veces la
# seña bien hecha no pasa su umbral sólo por dónde cayó la ventana.
#
# Entrenar también con la seña corrida es enseñarle el encuadre que va a ver.
DESPLAZAMIENTOS = (-9, -5, 5, 9)


def desplazar(seq, reposo, pasos):
    """La seña corrida `pasos` dentro de la ventana, rellenando con reposo."""
    if pasos == 0:
        return seq.copy()
    if pasos > 0:
        return np.concatenate([reposo[-pasos:], seq[:-pasos]])
    return np.concatenate([seq[-pasos:], reposo[:-pasos]])


def con_encuadres(X, y, reposos, rng, pasos=DESPLAZAMIENTOS):
    """
    Agrega, por cada secuencia, copias corridas dentro de la ventana.

    `reposos` son secuencias de la clase de reposo, de donde sale el relleno.
    Sin ellas no se puede rellenar con algo realista y se devuelve todo igual.
    """
    if not len(reposos):
        return X, y
    extra_x, extra_y = [], []
    for seq, etiqueta in zip(X, y):
        for k in pasos:
            extra_x.append(desplazar(seq, reposos[rng.integers(len(reposos))], k))
            extra_y.append(etiqueta)
    return (np.concatenate([X, np.array(extra_x)]),
            np.concatenate([y, np.array(extra_y)]))
