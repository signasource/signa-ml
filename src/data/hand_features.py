"""
Features de MANO para clasificación de señas estáticas (abecedario LSA).

Por qué no reusamos src/data/extractor.py + normalization.py:
  ese pipeline está pensado para señas dinámicas y normaliza TODO contra el
  ancho de hombros. Para el abecedario la seña ES la forma de la mano: el
  cuerpo no aporta nada y la escala de hombros mete ruido (si la persona está
  más cerca o de costado, los mismos dedos dan números distintos). Acá
  normalizamos contra la propia mano.

El vector final tiene 6 bloques, pensados para que letras parecidas
(A/E/S/M/N/T, U/V, D/G, etc.) queden separadas:

  A) 63  landmarks normalizados CONSERVANDO orientación (traslación + escala).
         Necesario: hay letras que sólo se diferencian por hacia dónde
         apunta la mano, no por la forma de los dedos.
  B) 63  landmarks en el marco canónico de la mano (traslación + escala +
         rotación 3D). Describe la forma pura, invariante a cómo esté girada.
  C)  9  los tres ejes del marco canónico → describe la orientación de la
         mano respecto de la cámara, explícita y compacta.
  D) 45  distancias entre los 10 puntos clave (muñeca, 4 nudillos, 5 yemas)
         sobre coordenadas canónicas. Captura "yema tocando yema", puño
         cerrado, etc., que es exactamente lo que distingue A de E o M de N.
  E) 15  cosenos de los ángulos de flexión de cada articulación (3 por dedo).
  F) 63  landmarks "world" de MediaPipe (métricos, en metros, centrados en la
         mano) llevados al mismo marco canónico. Vienen de otra cabeza del
         modelo y aportan profundidad más confiable que la z de imagen.

  Total: 258 valores.
"""
from __future__ import annotations

import numpy as np

# ─── Índices de landmarks de MediaPipe Hands ─────────────────────────────────
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20

# Cadenas de 5 puntos por dedo (muñeca → yema); los 3 puntos del medio son las
# articulaciones donde medimos flexión.
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],       # pulgar
    [0, 5, 6, 7, 8],       # índice
    [0, 9, 10, 11, 12],    # mayor
    [0, 13, 14, 15, 16],   # anular
    [0, 17, 18, 19, 20],   # meñique
]

# Puntos entre los que calculamos todas las distancias par a par (10 → 45 pares)
KEY_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP,
              THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

N_LANDMARKS = 21
FEATURE_DIM = 63 + 63 + 9 + 45 + 15 + 63  # 258

# Nombres de los bloques, en orden, para poder inspeccionar/depurar.
FEATURE_BLOCKS = [
    ("image_oriented", 63),
    ("canonical", 63),
    ("frame_axes", 9),
    ("pair_distances", 45),
    ("joint_angles", 15),
    ("world_canonical", 63),
]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)


def hand_frame(lm: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Construye un marco ortonormal propio de la mano.

    e_y  : muñeca → nudillo del dedo mayor ("hacia dónde apuntan los dedos")
    e_x  : a lo ancho de la palma (nudillo del índice → nudillo del meñique),
           ortogonalizado contra e_y
    e_z  : normal a la palma = e_x × e_y

    Devuelve (axes 3x3 con los ejes como filas, escala).
    La escala es el largo muñeca→nudillo del mayor: es el segmento más estable
    de la mano (no se acorta al flexionar los dedos), así que sirve de regla.
    """
    v_up = lm[MIDDLE_MCP] - lm[WRIST]
    scale = float(np.linalg.norm(v_up))
    if scale < 1e-8:
        return np.eye(3), 1.0

    e_y = v_up / scale
    v_across = lm[PINKY_MCP] - lm[INDEX_MCP]
    e_x = _unit(v_across - np.dot(v_across, e_y) * e_y)
    if np.allclose(e_x, 0):
        # Palma vista exactamente de canto: elegimos cualquier perpendicular.
        e_x = _unit(np.cross(e_y, np.array([0.0, 0.0, 1.0])))
    e_z = np.cross(e_x, e_y)

    return np.stack([e_x, e_y, e_z]), scale


def canonical_coords(lm: np.ndarray) -> np.ndarray:
    """Landmarks en el marco de la mano: invariantes a posición, escala y rotación."""
    axes, scale = hand_frame(lm)
    centered = lm - lm[WRIST]
    return (centered @ axes.T) / scale


def oriented_coords(lm: np.ndarray) -> np.ndarray:
    """Landmarks centrados en la muñeca y escalados, pero SIN rotar: conserva orientación."""
    _, scale = hand_frame(lm)
    return (lm - lm[WRIST]) / scale


def pair_distances(coords: np.ndarray) -> np.ndarray:
    """Distancias par a par entre los 10 puntos clave (45 valores)."""
    pts = coords[KEY_POINTS]
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(len(KEY_POINTS), k=1)
    return dists[iu]


def joint_angles(coords: np.ndarray) -> np.ndarray:
    """
    Coseno del ángulo en cada articulación intermedia de cada dedo (15 valores).
    1.0 = dedo estirado, valores negativos = muy flexionado.
    """
    out = []
    for chain in FINGER_CHAINS:
        for i in range(1, 4):
            a = coords[chain[i - 1]] - coords[chain[i]]
            b = coords[chain[i + 1]] - coords[chain[i]]
            out.append(-float(np.dot(_unit(a), _unit(b))))
    return np.array(out)


def mirror(lm: np.ndarray) -> np.ndarray:
    """Espeja la mano en x (izquierda ↔ derecha)."""
    out = lm.copy()
    out[:, 0] *= -1.0
    return out


def build_features(landmarks: np.ndarray,
                   world_landmarks: np.ndarray | None = None) -> np.ndarray:
    """
    Arma el vector de 258 features a partir de los landmarks de una mano.

    Args:
        landmarks: (21, 3) coordenadas normalizadas de imagen (x, y en [0,1], z relativa)
        world_landmarks: (21, 3) coordenadas métricas de MediaPipe, o None

    Returns:
        (258,) float32
    """
    lm = np.asarray(landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)

    axes, _ = hand_frame(lm)
    canon = canonical_coords(lm)

    blocks = [
        oriented_coords(lm).ravel(),   # A
        canon.ravel(),                 # B
        axes.ravel(),                  # C
        pair_distances(canon),         # D
        joint_angles(canon),           # E
    ]

    if world_landmarks is not None:
        wlm = np.asarray(world_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
        blocks.append(canonical_coords(wlm).ravel())  # F
    else:
        blocks.append(np.zeros(63))

    features = np.concatenate(blocks).astype(np.float32)
    assert features.shape == (FEATURE_DIM,), features.shape
    return features


def augment_features(features: np.ndarray, noise_std: float,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Ruido gaussiano leve sobre las features ya calculadas.

    Simula el jitter de detección de MediaPipe frame a frame, que es el ruido
    que el modelo va a ver en vivo. Se aplica en cada epoch, no se materializa
    en disco.
    """
    return features + rng.normal(0.0, noise_std, size=features.shape).astype(np.float32)


# ─── Bloque opcional: dónde está la mano respecto de la CARA ─────────────────
#
# Los 258 valores de arriba describen la mano y nada más: están normalizados
# contra la propia mano, así que la ubicación del brazo se pierde a propósito.
# Para casi todo el abecedario eso está bien, pero hay letras cuya forma es
# idéntica y sólo se distinguen por dónde se apoya la mano — T en la pera e I
# en el ojo son el caso claro. Sin esto, el modelo no puede separarlas: no es
# que le falten datos, es que la diferencia no está en su entrada.
#
# La referencia es la cara del propio señante (ojos y boca), así que el valor
# no depende de cuán lejos esté de la cámara ni de dónde esté parado.

FACE_BLOCK_DIM = 8
FACE_BLOCK_NAME = "face_relative"

# Posiciones dentro del bloque, para no andar con índices sueltos por ahí.
FACE_WRIST_X, FACE_WRIST_Y = 0, 1
FACE_INDEX_X, FACE_INDEX_Y = 2, 3
FACE_MIDDLE_X, FACE_MIDDLE_Y = 4, 5
FACE_HAND_SIZE, FACE_PRESENT = 6, 7


def build_face_features(landmarks: np.ndarray,
                        eye_center: np.ndarray | None,
                        mouth_center: np.ndarray | None,
                        mirrored: bool = False) -> np.ndarray:
    """
    Ubicación de la mano en coordenadas de la cara.

    Args:
        landmarks: (21, 3) landmarks de imagen SIN espejar (x, y en [0,1]).
        eye_center: (2,) punto medio entre los ojos, o None si no se detectó cara.
        mouth_center: (2,) centro de la boca, o None.
        mirrored: True si la mano se espejó para tratarla como derecha; en ese
            caso hay que espejar también el eje x de este bloque, o el mismo
            gesto daría signos opuestos según con qué mano se haga.

    Returns:
        (8,) float32 — 3 puntos de la mano en coordenadas de cara (x, y),
        el tamaño de la mano relativo al de la cara, y un flag de presencia.
        Todo en ceros con flag 0 si no hay cara: el modelo aprende a ignorarlo.
    """
    if eye_center is None or mouth_center is None:
        return np.zeros(FACE_BLOCK_DIM, dtype=np.float32)

    lm = np.asarray(landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)[:, :2]
    ojo = np.asarray(eye_center, dtype=np.float64).reshape(2)
    boca = np.asarray(mouth_center, dtype=np.float64).reshape(2)

    # Escala propia de la cara: la distancia ojos–boca. Invariante a la distancia
    # a la cámara, y mucho más estable que el ancho del bounding box.
    escala = float(np.linalg.norm(boca - ojo))
    if escala < 1e-6:
        return np.zeros(FACE_BLOCK_DIM, dtype=np.float32)

    out = []
    for idx in (WRIST, INDEX_TIP, MIDDLE_TIP):
        d = (lm[idx] - ojo) / escala
        out += [-d[0] if mirrored else d[0], d[1]]

    # Tamaño de la mano contra el de la cara: separa gestos cerca de la cámara
    # de gestos pegados a la cara, que en 2D caen en el mismo lugar.
    mano = float(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST])) / escala
    out += [mano, 1.0]                      # último valor: flag "hay cara"
    return np.asarray(out, dtype=np.float32)


# ─── Aumentación en el espacio de features ───────────────────────────────────
#
# Las fotos del dataset muestran cada letra como la hizo esa persona en ese
# video: la A casi siempre de costado y a la altura del hombro. Con 15 fotos
# por letra el modelo aprende eso como parte de la letra, y una A de frente o
# más abajo deja de pasar. Las dos funciones de acá le muestran la misma mano
# en otras posturas sin tener que volver a correr MediaPipe.

# Bloques que dependen de hacia dónde mira la mano. El resto (canónicas,
# distancias, ángulos, world canónicas) ya es invariante a la rotación.
_ORIENTED = slice(0, 63)
_AXES = slice(126, 135)


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """
    Rotación en ejes de imagen (x derecha, y abajo, z profundidad).

    yaw   gira alrededor del eje vertical: es pasar de mostrar la mano de frente
          a mostrarla de costado.
    pitch la inclina hacia la cámara o hacia atrás.
    roll  la gira en el plano de la imagen (la aumentación de imagen ya cubre
          ±12°, así que suele ir en cero).
    """
    a, b, c = np.radians([yaw_deg, pitch_deg, roll_deg])
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ rx @ ry


def rotate_hand_features(features: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Las features de la misma mano vista girada R.

    Rotar los landmarks y recalcular da exactamente esto: las coordenadas
    orientadas (bloque A) y los ejes del marco de la mano (bloque C) giran con
    R, y todo lo que está en el marco propio de la mano no cambia. Acepta un
    vector o una matriz de filas; las columnas después de las 258 de mano (el
    bloque de cara) se dejan como están.
    """
    out = np.array(features, dtype=np.float32, copy=True)
    flat = out.reshape(-1, out.shape[-1])
    n = flat.shape[0]
    flat[:, _ORIENTED] = (flat[:, _ORIENTED].reshape(n, 21, 3) @ R.T).reshape(n, 63)
    flat[:, _AXES] = (flat[:, _AXES].reshape(n, 3, 3) @ R.T).reshape(n, 9)
    return flat.reshape(out.shape)


def shift_face_block(face: np.ndarray, dx: float, dy: float, size: float) -> np.ndarray:
    """
    El bloque de cara de la misma mano corrida (dx, dy) en unidades de
    ojos-a-boca y con su tamaño relativo multiplicado por `size` (más cerca o
    más lejos de la cámara). Un bloque sin cara (flag en 0) se deja igual.
    """
    out = np.array(face, dtype=np.float32, copy=True)
    if out[FACE_PRESENT] != 1.0:
        return out
    for i in (FACE_WRIST_X, FACE_INDEX_X, FACE_MIDDLE_X):
        out[i] += dx
    for i in (FACE_WRIST_Y, FACE_INDEX_Y, FACE_MIDDLE_Y):
        out[i] += dy
    out[FACE_HAND_SIZE] *= size
    return out
