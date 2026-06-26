"""
Extractor de landmarks usando MediaPipe Holistic.

Usa el modelo Holistic (pose + manos) en lugar del detector de manos solo.
(posición de los brazos, distancia al cuerpo, etc.).
"""
import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def extract_keypoints(results) -> np.ndarray:
    """
    Convierte los resultados de MediaPipe Holistic a un vector plano de features.

    Estructura:
      - pose:        33 puntos × 4 valores (x, y, z, visibility) = 132
      - left_hand:   21 puntos × 3 valores (x, y, z)             =  63
      - right_hand:  21 puntos × 3 valores (x, y, z)             =  63
      Total: 258 valores por frame.

    Si un landmark group no se detecta, se rellena con ceros.
    """
    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    lh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, lh, rh])  # shape: (258,)


def draw_landmarks(image, results):
    """Dibuja los landmarks sobre el frame para visualización."""
    # Pose
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    # Mano izquierda
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
    # Mano derecha
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
    return image


def process_frame(frame, holistic_model):
    """
    Procesa un frame BGR de OpenCV con MediaPipe Holistic.
    Retorna (frame_con_dibujo, results, keypoints_array).
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    draw_landmarks(image, results)
    keypoints = extract_keypoints(results)
    return image, results, keypoints


def get_holistic_model(min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
    """Devuelve una instancia del modelo Holistic lista para usar."""
    return mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
