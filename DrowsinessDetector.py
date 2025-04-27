import queue
import threading
import time
import winsound
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import sys
import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QHBoxLayout, 
                            QWidget, QPushButton, QVBoxLayout, QComboBox, QMessageBox,
                            QFrame, QGridLayout)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QTimer

class DrowsinessDetector(QMainWindow):
    def __init__(self):
        super().__init__()

        self.yawn_state = ''
        self.left_eye_state =''
        self.right_eye_state= ''
        self.alert_text = ''

        self.blinks = 0
        self.microsleeps = 0
        self.yawns = 0
        self.yawn_duration = 0 

        self.left_eye_still_closed = False  
        self.right_eye_still_closed = False 
        self.yawn_in_progress = False  
        
        # Initialisation de MediaPipe Face Mesh avec plus de points
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1,
            refine_landmarks=True
        )
        
        # Points MediaPipe pour la bouche et les yeux (pour extraction des ROI)
        # Bouche : points 61 (gauche), 291 (droite), 0 (haut), 17 (bas)
        self.mouth_roi_points = [61, 291, 0, 17]
        # Oeil droit : 33 (gauche), 133 (droite), 159 (haut), 145 (bas)
        self.right_eye_roi_points = [33, 133, 159, 145]
        # Oeil gauche : 362 (gauche), 263 (droite), 386 (haut), 374 (bas)
        self.left_eye_roi_points = [362, 263, 386, 374]
        
        # Arduino setup
        self.arduino = None
        self.arduino_ports = []
        self.refresh_ports()

        self.setWindowTitle("Smart Drowsiness Detection System")
        self.setGeometry(100, 50, 1400, 900)
        
        # Style moderne et professionnel
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #1a1a2e; 
            }
            QLabel { 
                font-family: 'Segoe UI', Arial; 
                color: #ffffff;
            }
            QPushButton { 
                background-color: #4a4e69; 
                color: white; 
                padding: 10px 20px; 
                border-radius: 8px; 
                font-weight: bold;
                font-size: 14px;
                margin: 5px;
            }
            QPushButton:hover { 
                background-color: #6c6d8c; 
            }
            QPushButton:pressed {
                background-color: #3a3d5c;
            }
            QComboBox { 
                padding: 8px; 
                border: 2px solid #4a4e69; 
                border-radius: 8px; 
                background: #2d2d3d;
                color: white;
                min-width: 150px;
            }
            QFrame {
                background-color: #16213e;
                border-radius: 15px;
                border: 2px solid #4a4e69;
            }
        """)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Left panel for video and controls
        self.left_panel = QFrame()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # Controls panel
        self.controls_widget = QFrame()
        self.controls_layout = QHBoxLayout(self.controls_widget)
        
        # --- Sélection de la caméra ---
        self.available_cameras = self.get_available_cameras()
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Camera {i}" for i in self.available_cameras])
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        self.controls_layout.insertWidget(0, self.camera_combo)
        self.current_camera_index = self.available_cameras[0] if self.available_cameras else 0
        self.cap = cv2.VideoCapture(self.current_camera_index)
        time.sleep(1.000)
        
        # Port selection
        self.port_combo = QComboBox()
        self.port_combo.addItems(self.arduino_ports)
        self.controls_layout.addWidget(self.port_combo)
        
        # Refresh ports button
        self.refresh_btn = QPushButton("🔄 Refresh Ports")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.controls_layout.addWidget(self.refresh_btn)
        
        # Connect button
        self.connect_btn = QPushButton("🔌 Connect Arduino")
        self.connect_btn.clicked.connect(self.connect_arduino)
        self.controls_layout.addWidget(self.connect_btn)
        
        # Quit button
        self.quit_btn = QPushButton("Quit")
        self.quit_btn.setStyleSheet("background-color: #e11d48; color: white; font-weight: bold;")
        self.quit_btn.clicked.connect(self.close)
        self.controls_layout.addWidget(self.quit_btn)
        
        self.left_layout.addWidget(self.controls_widget)

        # Video display
        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("""
            border: 3px solid #4a4e69; 
            border-radius: 15px; 
            padding: 5px;
            background-color: #2d2d3d;
            margin: 10px;
        """)
        self.video_label.setFixedSize(800, 600)
        self.left_layout.addWidget(self.video_label)
        
        self.main_layout.addWidget(self.left_panel)

        # Right panel for information
        self.right_panel = QFrame()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        # Status indicators
        self.status_frame = QFrame()
        self.status_layout = QGridLayout(self.status_frame)
        
        # Eye status
        self.eye_status = QLabel("👁️ Eye Status: Normal")
        self.eye_status.setStyleSheet("font-size: 16px; color: #4ecca3;")
        self.status_layout.addWidget(self.eye_status, 0, 0)
        
        # Mouth status
        self.mouth_status = QLabel("👄 Mouth Status: Normal")
        self.mouth_status.setStyleSheet("font-size: 16px; color: #4ecca3;")
        self.status_layout.addWidget(self.mouth_status, 1, 0)
        
        # Alert status
        self.alert_status = QLabel("⚠️ Alert Status: None")
        self.alert_status.setStyleSheet("font-size: 16px; color: #4ecca3;")
        self.status_layout.addWidget(self.alert_status, 2, 0)
        
        self.right_layout.addWidget(self.status_frame)
        
        # Statistics
        self.stats_frame = QFrame()
        self.stats_layout = QGridLayout(self.stats_frame)
        
        self.blink_count = QLabel("Blinks: 0")
        self.blink_count.setStyleSheet("font-size: 14px;")
        self.stats_layout.addWidget(self.blink_count, 0, 0)
        
        self.yawn_count = QLabel("Yawns: 0")
        self.yawn_count.setStyleSheet("font-size: 14px;")
        self.stats_layout.addWidget(self.yawn_count, 0, 1)
        
        self.microsleep_time = QLabel("Microsleep: 0.0s")
        self.microsleep_time.setStyleSheet("font-size: 14px;")
        self.stats_layout.addWidget(self.microsleep_time, 1, 0)
        
        self.yawn_duration_label = QLabel("Yawn Duration: 0.0s")
        self.yawn_duration_label.setStyleSheet("font-size: 14px;")
        self.stats_layout.addWidget(self.yawn_duration_label, 1, 1)
        
        self.right_layout.addWidget(self.stats_frame)
        
        self.main_layout.addWidget(self.right_panel)

        # Initialize detection models
        self.detectyawn = YOLO("runs/detectyawn/train/weights/best.pt")
        self.detecteye = YOLO("runs/detecteye/train/weights/best.pt")
        
        # Initialize frame processing
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)

        self.capture_thread.start()
        self.process_thread.start()
        
        # Timer for updating UI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)  # Update every 100ms

        # Pour la détection orientation tête
        self.head_pose_alert = False
        self.head_pose_timer = 0
        self.head_pose_threshold = 2.0  # secondes
        self.last_head_pose_time = time.time()
        self.head_pose_limit = 20  # degrés (à ajuster)

    def update_ui(self):
        """Update UI elements"""
        self.blink_count.setText(f"Blinks: {self.blinks}")
        self.yawn_count.setText(f"Yawns: {self.yawns}")
        self.microsleep_time.setText(f"Microsleep: {round(self.microsleeps, 2)}s")
        self.yawn_duration_label.setText(f"Yawn Duration: {round(self.yawn_duration, 2)}s")
        
        # Update status indicators
        if self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye":
            self.eye_status.setText("👁️ Eye Status: Closed")
            self.eye_status.setStyleSheet("font-size: 16px; color: #ff6b6b;")
        else:
            self.eye_status.setText("👁️ Eye Status: Open")
            self.eye_status.setStyleSheet("font-size: 16px; color: #4ecca3;")
            
        if self.yawn_state == "Yawn":
            self.mouth_status.setText("👄 Mouth Status: Yawning")
            self.mouth_status.setStyleSheet("font-size: 16px; color: #ff6b6b;")
        else:
            self.mouth_status.setText("👄 Mouth Status: Normal")
            self.mouth_status.setStyleSheet("font-size: 16px; color: #4ecca3;")
            
        if self.alert_text:
            self.alert_status.setText("⚠️ Alert Status: Active")
            self.alert_status.setStyleSheet("font-size: 16px; color: #ff6b6b;")
        else:
            self.alert_status.setText("⚠️ Alert Status: None")
            self.alert_status.setStyleSheet("font-size: 16px; color: #4ecca3;")

    def process_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = None
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                if frame is not None and ret:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(image_rgb)

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            self.draw_face_landmarks(frame, face_landmarks)
                            self.process_detection(frame, face_landmarks)
                    # Toujours afficher la frame, même sans détection
                    self.display_frame(frame)
                else:
                    # Afficher un écran noir si la caméra ne fournit pas d'image
                    black = np.zeros((600, 800, 3), dtype=np.uint8)
                    self.display_frame(black)
                    time.sleep(0.1)
            except Exception as e:
                print(f"Erreur dans process_frames : {e}")
                black = np.zeros((600, 800, 3), dtype=np.uint8)
                self.display_frame(black)
                time.sleep(0.1)

    def draw_face_landmarks(self, frame, face_landmarks):
        """Draw face landmarks on the frame"""
        ih, iw, _ = frame.shape
        
        # Draw eye points
        for point_id in self.left_eye_roi_points:
            lm = face_landmarks.landmark[point_id]
            x, y = int(lm.x * iw), int(lm.y * ih)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
        # Draw mouth points
        for point_id in self.mouth_roi_points:
            lm = face_landmarks.landmark[point_id]
            x, y = int(lm.x * iw), int(lm.y * ih)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    def process_detection(self, frame, face_landmarks):
        """Process drowsiness detection"""
        ih, iw, _ = frame.shape
        # --- Extraction des ROI avec les nouveaux points ---
        # Bouche
        x1, y1 = int(face_landmarks.landmark[self.mouth_roi_points[0]].x * iw), int(face_landmarks.landmark[self.mouth_roi_points[0]].y * ih)
        x2, y2 = int(face_landmarks.landmark[self.mouth_roi_points[1]].x * iw), int(face_landmarks.landmark[self.mouth_roi_points[1]].y * ih)
        y_top = int(face_landmarks.landmark[self.mouth_roi_points[2]].y * ih)
        y_bot = int(face_landmarks.landmark[self.mouth_roi_points[3]].y * ih)
        mouth_roi = frame[min(y_top, y1, y2):max(y_bot, y1, y2), min(x1, x2):max(x1, x2)]
        # Oeil droit
        rx1, ry1 = int(face_landmarks.landmark[self.right_eye_roi_points[0]].x * iw), int(face_landmarks.landmark[self.right_eye_roi_points[0]].y * ih)
        rx2, ry2 = int(face_landmarks.landmark[self.right_eye_roi_points[1]].x * iw), int(face_landmarks.landmark[self.right_eye_roi_points[1]].y * ih)
        r_top = int(face_landmarks.landmark[self.right_eye_roi_points[2]].y * ih)
        r_bot = int(face_landmarks.landmark[self.right_eye_roi_points[3]].y * ih)
        right_eye_roi = frame[min(r_top, ry1, ry2):max(r_bot, ry1, ry2), min(rx1, rx2):max(rx1, rx2)]
        # Oeil gauche
        lx1, ly1 = int(face_landmarks.landmark[self.left_eye_roi_points[0]].x * iw), int(face_landmarks.landmark[self.left_eye_roi_points[0]].y * ih)
        lx2, ly2 = int(face_landmarks.landmark[self.left_eye_roi_points[1]].x * iw), int(face_landmarks.landmark[self.left_eye_roi_points[1]].y * ih)
        l_top = int(face_landmarks.landmark[self.left_eye_roi_points[2]].y * ih)
        l_bot = int(face_landmarks.landmark[self.left_eye_roi_points[3]].y * ih)
        left_eye_roi = frame[min(l_top, ly1, ly2):max(l_bot, ly1, ly2), min(lx1, lx2):max(lx1, lx2)]
        # ---
        self.process_eye_detection(left_eye_roi, right_eye_roi)
        self.process_yawn_detection(mouth_roi)
        self.detect_head_pose(frame, face_landmarks)
        self.update_alerts()

    def process_eye_detection(self, left_eye_roi, right_eye_roi):
        """Process eye detection"""
        self.left_eye_state = self.predict_eye(left_eye_roi, self.left_eye_state)
        self.right_eye_state = self.predict_eye(right_eye_roi, self.right_eye_state)
        
        if self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye":
            if not self.left_eye_still_closed and not self.right_eye_still_closed:
                self.left_eye_still_closed, self.right_eye_still_closed = True, True
                self.blinks += 1
            self.microsleeps += 45 / 1000
        else:
            if self.left_eye_still_closed and self.right_eye_still_closed:
                self.left_eye_still_closed, self.right_eye_still_closed = False, False
            self.microsleeps = 0

    def process_yawn_detection(self, mouth_roi):
        """Process yawn detection"""
        self.predict_yawn(mouth_roi)
        
        if self.yawn_state == "Yawn":
            if not self.yawn_in_progress:
                self.yawn_in_progress = True
                self.yawns += 1
            self.yawn_duration += 45 / 1000
        else:
            if self.yawn_in_progress:
                self.yawn_in_progress = False
                self.yawn_duration = 0

    def update_alerts(self):
        """Update alert status"""
        self.alert_text = ''
        
        if round(self.yawn_duration, 2) > 7.0:
            self.play_alert_sound()
            self.alert_text = "⚠️ Prolonged Yawn Detected!"
            
        if round(self.microsleeps, 2) > 4.0:
            self.play_alert_sound()
            self.alert_text = "⚠️ Prolonged Microsleep Detected!"

    def play_alert_sound(self):
        """Play alert sound and trigger buzzer"""
        frequency = 1000
        duration = 500
        winsound.Beep(frequency, duration)
        
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(b'D')
                time.sleep(0.5)
            except Exception as e:
                print(f"Error sending command to buzzer: {e}")

    def display_frame(self, frame):
        """Display processed frame"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def refresh_ports(self):
        """Refresh available serial ports et connecte automatiquement si possible"""
        self.arduino_ports = [port.device for port in serial.tools.list_ports.comports()]
        if hasattr(self, 'port_combo'):
            self.port_combo.clear()
            self.port_combo.addItems(self.arduino_ports)
            if self.arduino_ports:
                self.port_combo.setCurrentIndex(0)
                self.connect_arduino()
    
    def connect_arduino(self):
        """Connecte automatiquement à l'Arduino dès que possible"""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            self.connect_btn.setText("🔌 Disconnect Arduino")
            QMessageBox.information(self, "Success", f"Connected to Arduino on {self.port_combo.currentText()}")
        else:
            QMessageBox.warning(self, "Error", "No port selected!")

    def closeEvent(self, event):
        """Clean up resources when closing"""
        self.stop_event.set()
        if self.capture_thread.is_alive():
            self.capture_thread.join()
        if self.process_thread.is_alive():
            self.process_thread.join()
        if self.cap.isOpened():
            self.cap.release()
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
        event.accept()

    def capture_frames(self):
        """Capture frames from the camera"""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(frame)
            else:
                break

    def predict_eye(self, eye_frame, eye_state):
        """Predict eye state using YOLO model"""
        results_eye = self.detecteye.predict(eye_frame)
        boxes = results_eye[0].boxes
        if len(boxes) == 0:
            return eye_state

        confidences = boxes.conf.cpu().numpy()  
        class_ids = boxes.cls.cpu().numpy()  
        max_confidence_index = np.argmax(confidences)
        class_id = int(class_ids[max_confidence_index])

        if class_id == 1:
            eye_state = "Close Eye"
        elif class_id == 0 and confidences[max_confidence_index] > 0.30:
            eye_state = "Open Eye"
                            
        return eye_state

    def predict_yawn(self, yawn_frame):
        """Predict yawn state using YOLO model"""
        results_yawn = self.detectyawn.predict(yawn_frame)
        boxes = results_yawn[0].boxes

        if len(boxes) == 0:
            return self.yawn_state

        confidences = boxes.conf.cpu().numpy()  
        class_ids = boxes.cls.cpu().numpy()  
        max_confidence_index = np.argmax(confidences)
        class_id = int(class_ids[max_confidence_index])

        if class_id == 0:
            self.yawn_state = "Yawn"
        elif class_id == 1 and confidences[max_confidence_index] > 0.50:
            self.yawn_state = "No Yawn"

    def get_available_cameras(self, max_tested=5):
        """Détecte les caméras disponibles (0 à max_tested-1)"""
        available = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def change_camera(self, index):
        """Change la caméra utilisée pour le flux vidéo"""
        new_index = self.available_cameras[index]
        if self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(new_index)
        self.current_camera_index = new_index
        time.sleep(1.000)
        # Vérifier si la caméra s'ouvre
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Erreur caméra", f"Impossible d'ouvrir la caméra {new_index}. Branchez-la ou essayez un autre port.")

    def detect_head_pose(self, frame, face_landmarks):
        """Détecte l'orientation de la tête (pitch/yaw/roll) et déclenche une alerte si besoin"""
        ih, iw, _ = frame.shape
        # Points clés pour l'estimation (nez, menton, yeux, bouche)
        image_points = np.array([
            (int(face_landmarks.landmark[1].x * iw), int(face_landmarks.landmark[1].y * ih)),     # Nose tip
            (int(face_landmarks.landmark[152].x * iw), int(face_landmarks.landmark[152].y * ih)), # Chin
            (int(face_landmarks.landmark[263].x * iw), int(face_landmarks.landmark[263].y * ih)), # Right eye right corner
            (int(face_landmarks.landmark[33].x * iw), int(face_landmarks.landmark[33].y * ih)),   # Left eye left corner
            (int(face_landmarks.landmark[287].x * iw), int(face_landmarks.landmark[287].y * ih)), # Mouth right
            (int(face_landmarks.landmark[57].x * iw), int(face_landmarks.landmark[57].y * ih)),   # Mouth left
        ], dtype='double')
        # Modèle 3D simplifié du visage
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -63.6, -12.5),         # Chin
            (43.3, 32.7, -26.0),         # Right eye right corner
            (-43.3, 32.7, -26.0),        # Left eye left corner
            (28.9, -28.9, -24.1),        # Mouth right
            (-28.9, -28.9, -24.1)        # Mouth left
        ])
        # Camera internals
        focal_length = iw
        center = (iw / 2, ih / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype='double')
        dist_coeffs = np.zeros((4, 1))
        # Résolution PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return
        # Conversion en angles d'Euler
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
        # Affichage sur la frame
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Roll: {roll:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        # Détection d'une position anormale
        if abs(pitch) > self.head_pose_limit or abs(yaw) > self.head_pose_limit:
            if not self.head_pose_alert:
                self.last_head_pose_time = time.time()
                self.head_pose_alert = True
            elif time.time() - self.last_head_pose_time > self.head_pose_threshold:
                # Alerte : tête penchée trop longtemps
                cv2.putText(frame, "ALERTE: TETE PENCHEE!", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                self.play_alert_sound()
        else:
            self.head_pose_alert = False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessDetector()
    window.show()
    sys.exit(app.exec_())