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
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = frame.shape
            # (ajoute ici ta logique de détection, points, angles, etc.)
            # Exemple : dessiner un point sur le nez
            lm = face_landmarks.landmark[1]
            x, y = int(lm.x * iw), int(lm.y * ih)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

demo = gr.Interface(
    fn=detect_drowsiness,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs="image",
    live=True,
    title="Détection de Somnolence (Webcam)"
)

if __name__ == "__main__":
    demo.launch()