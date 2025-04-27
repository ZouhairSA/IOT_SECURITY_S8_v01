import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

st.set_page_config(page_title="Détection de Somnolence", layout="centered")
st.title("🛡️ Détection de Somnolence (Démo Web)")

# Chargement des modèles
@st.cache_resource
def load_models():
    detectyawn = YOLO("runs/detectyawn/train/weights/best.pt")
    detecteye = YOLO("runs/detecteye/train/weights/best.pt")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_faces=1,
        refine_landmarks=True
    )
    return detectyawn, detecteye, face_mesh

detectyawn, detecteye, face_mesh = load_models()

uploaded_file = st.file_uploader("Choisissez une image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = frame.shape

            # Points pour la bouche et les yeux
            mouth_points = [61, 291, 0, 17]
            right_eye_points = [33, 133, 159, 145]
            left_eye_points = [362, 263, 386, 374]

            # Dessiner les points
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

            # --- Détection yeux et bouche (YOLO) ---
            # (exemple pour la bouche, à adapter pour les yeux)
            # mouth_roi = ... (extraction comme dans ton code desktop)
            # results_yawn = detectyawn.predict(mouth_roi)
            # results_eye = detecteye.predict(eye_roi)
            # st.write('Résultat de la détection : ...')

    st.image(frame, channels="BGR", caption="Résultat de la détection")

else:
    st.info("Veuillez uploader une image pour commencer.")

st.markdown("---")
st.markdown("**Démo web basée sur la logique du projet desktop.**")
