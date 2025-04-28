from datetime import datetime
import time
import cv2
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk
from database import MongoDBConnection

class CaptureCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Player")

        # Initialisation de la connexion à MongoDB
        self.db = MongoDBConnection()

        self.cap = cv2.VideoCapture(0)
        time.sleep(1.000)
        self.label = Label(root)
        self.label.grid(column=0, row=3, columnspan=3)

        self.btn_yawn = Button(root, text="Capture for yawing", command=lambda: self.capture_action_image("yawn"))
        self.btn_yawn.grid(column=0, row=0, padx=5, pady=5)

        self.btn_drowsy = Button(root, text="Capture for closed eyes", command=lambda: self.capture_action_image("drowsy"))
        self.btn_drowsy.grid(column=1, row=0, padx=5, pady=5)

        self.btn_awake = Button(root, text="Capture for open eyes", command=lambda: self.capture_action_image("awake"))
        self.btn_awake.grid(column=2, row=0, padx=5, pady=5)

        self.lblinfo = Label(root, text="Video input: ")
        self.lblinfo.grid(column=1, row=1, rowspan=1)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
            self.label.after(10, self.update_frame)
        else:
            self.cap.release()
            self.db.close()

    def capture_action_image(self, action):
        ret, frame = self.cap.read()
        if ret:
            dateid = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            image_path = f"database/untagged_images/{action}_{dateid}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image captured and saved as '{image_path}'")

            # Enregistrer les métadonnées dans MongoDB
            image_data = {
                "action": action,
                "timestamp": dateid,
                "image_path": image_path
            }
            self.db.insert_data("captured_images", image_data)

root = tk.Tk()
app = CaptureCameraApp(root)
root.mainloop()