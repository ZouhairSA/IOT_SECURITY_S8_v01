import gradio as gr
import cv2
import numpy as np
import os
from ultralytics import YOLO
import mediapipe as mp
import time

# Chemins des modèles
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yawn_model_path = os.path.join(BASE_DIR, "runs", "detectyawn", "train", "weights", "best.pt")
# eye_model_path = os.path.join(BASE_DIR, "runs", "detecteye", "train", "weights", "best.pt")

# Chargement des modèles
detectyawn = YOLO(yawn_model_path)
# detecteye = YOLO(eye_model_path)
face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1,
    refine_landmarks=True
)

# Seuils pour la détection
HEAD_POSE_LIMIT = 20  # degrés
YAWN_DURATION_LIMIT = 7.0  # secondes
MICROSLEEP_LIMIT = 4.0  # secondes

# Variables d'état globales (pour la session Gradio)
yawn_in_progress = False
yawn_duration = 0
left_eye_still_closed = False
right_eye_still_closed = False
microsleeps = 0
last_head_pose_time = time.time()
head_pose_alert = False

def detect_drowsiness(image):
    global yawn_in_progress, yawn_duration, left_eye_still_closed, right_eye_still_closed, microsleeps, last_head_pose_time, head_pose_alert

    if image is None:
        return None, "Aucune image"

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    alert_text = ""

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = frame.shape

            # Points pour la bouche et les yeux
            mouth_points = [61, 291, 0, 17]
            right_eye_points = [33, 133, 159, 145]
            left_eye_points = [362, 263, 386, 374]
            for pt in mouth_points + right_eye_points + left_eye_points:
                lm = face_landmarks.landmark[pt]
                x, y = int(lm.x * iw), int(lm.y * ih)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # --- Détection orientation tête (pitch/yaw/roll) ---
            image_points = np.array([
                (int(face_landmarks.landmark[1].x * iw), int(face_landmarks.landmark[1].y * ih)),     # Nose tip
                (int(face_landmarks.landmark[152].x * iw), int(face_landmarks.landmark[152].y * ih)), # Chin
                (int(face_landmarks.landmark[263].x * iw), int(face_landmarks.landmark[263].y * ih)), # Right eye right corner
                (int(face_landmarks.landmark[33].x * iw), int(face_landmarks.landmark[33].y * ih)),   # Left eye left corner
                (int(face_landmarks.landmark[287].x * iw), int(face_landmarks.landmark[287].y * ih)), # Mouth right
                (int(face_landmarks.landmark[57].x * iw), int(face_landmarks.landmark[57].y * ih)),   # Mouth left
            ], dtype='double')
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -63.6, -12.5),         # Chin
                (43.3, 32.7, -26.0),         # Right eye right corner
                (-43.3, 32.7, -26.0),        # Left eye left corner
                (28.9, -28.9, -24.1),        # Mouth right
                (-28.9, -28.9, -24.1)        # Mouth left
            ])
            focal_length = iw
            center = (iw / 2, ih / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype='double')
            dist_coeffs = np.zeros((4, 1))
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                rmat, _ = cv2.Rodrigues(rotation_vector)
                sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
                singular = sy < 1e-6
                if not singular:
                    x = np.arctan2(rmat[2, 1], rmat[2, 2])
                    y = np.arctan2(-rmat[2, 0], sy)
                    z = np.arctan2(rmat[1, 0], rmat[0, 0])
                else:
                    x = np.arctan2(-rmat[1, 2], rmat[1, 1])
                    y = np.arctan2(-rmat[2, 0], sy)
                    z = 0
                pitch = np.degrees(x)
                yaw = np.degrees(y)
                roll = np.degrees(z)
                cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(frame, f"Roll: {roll:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                # Détection d'une position anormale
                if abs(pitch) > HEAD_POSE_LIMIT or abs(yaw) > HEAD_POSE_LIMIT:
                    if not head_pose_alert:
                        last_head_pose_time = time.time()
                        head_pose_alert = True
                    elif time.time() - last_head_pose_time > 2.0:
                        cv2.putText(frame, "ALERTE: TETE PENCHEE!", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        alert_text += "⚠️ Tête penchée détectée !\n"
                else:
                    head_pose_alert = False

            # --- Détection bouche/yeux (exemple simple, à enrichir avec YOLO si besoin) ---
            # Tu peux ajouter ici la logique de détection avancée (bâillement, yeux fermés, etc.)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), alert_text if alert_text else "Aucune alerte"

demo = gr.Interface(
    fn=detect_drowsiness,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Webcam"),
    outputs=[gr.Image(label="Résultat"), gr.Textbox(label="Alerte")],
    live=True,
    title="Détection de Somnolence (WebCam Live)",
    description="Testez la détection de somnolence en direct avec votre webcam."
)

if __name__ == "__main__":
    demo.launch()