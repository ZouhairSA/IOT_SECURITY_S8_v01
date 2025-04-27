import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Charger les modèles
detectyawn = YOLO("runs/detectyawn/train/weights/best.pt")
detecteye = YOLO("runs/detecteye/train/weights/best.pt")
face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1,
    refine_landmarks=True
)

def detect_drowsiness(image):
    if image is None:
        return None
    # Gradio fournit l'image en RGB uint8
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
            # (Tu peux ajouter ici la logique de détection yeux/bouche/orientation tête)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

demo = gr.Interface(
    fn=detect_drowsiness,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Webcam"),
    outputs=gr.Image(label="Résultat"),
    live=True,
    title="Détection de Somnolence (Webcam Live)",
    description="Testez la détection de somnolence en direct avec votre webcam."
)

if __name__ == "__main__":
    demo.launch()