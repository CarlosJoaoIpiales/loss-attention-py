import cv2
import dlib
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from scipy.spatial import distance as dist
import time

# Inicializar detector de rostros y predictor de rasgos faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Función para obtener la región de los ojos
def get_eye_region(landmarks, left=True):
    if left:
        points = [36, 37, 38, 39, 40, 41]
    else:
        points = [42, 43, 44, 45, 46, 47]
    region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
    return region

# Función para calcular el centroide de una región
def get_centroid(region):
    M = cv2.moments(region)
    if M["m00"] == 0:
        return 0, 0
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

# Función para calcular el Aspecto del Ojo (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Función para calcular el Aspecto de la Boca (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Función para determinar si hay pérdida de atención
def is_distracted(left_centroid, right_centroid, face_rect):
    face_center_x = (face_rect.left() + face_rect.right()) // 2
    face_center_y = (face_rect.top() + face_rect.bottom()) // 2
    threshold = 0.26
    left_dist = np.linalg.norm(np.array(left_centroid) - np.array([face_center_x, face_center_y]))
    right_dist = np.linalg.norm(np.array(right_centroid) - np.array([face_center_x, face_center_y]))
    return left_dist > face_rect.width() * threshold or right_dist > face_rect.width() * threshold

# Interfaz gráfica con tkinter
class AttentionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Pérdida de Atención")
        self.root.geometry("800x680")  # Tamaño fijo para la ventana

        # Frame para la visualización del video
        self.video_frame = tk.Frame(root, width=800, height=480)
        self.video_frame.pack(fill=tk.BOTH, expand=False)

        # Label para mostrar el video
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Frame para los botones
        self.button_frame = tk.Frame(root, width=800, height=120)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.start_button = tk.Button(self.button_frame, text="Iniciar captura", width=20, command=self.start_capture)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = tk.Button(self.button_frame, text="Detener captura", width=20, command=self.stop_capture)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.quit_button = tk.Button(self.button_frame, text="Salir", width=20, command=self.quit_app)
        self.quit_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Etiquetas para mostrar alertas y contadores
        self.alert_label = tk.Label(root, text="", font=("Arial", 24), fg="red")
        self.alert_label.pack()

        self.blink_label = tk.Label(root, text="Parpadeos: 0", font=("Arial", 24), fg="blue")
        self.blink_label.pack()

        self.yawn_label = tk.Label(root, text="Bostezos: 0", font=("Arial", 24), fg="purple")
        self.yawn_label.pack()

        self.sleepy_label = tk.Label(root, text="", font=("Arial", 24), fg="orange")
        self.sleepy_label.pack()

        self.cap = None
        self.is_capturing = False
        self.blink_count = 0
        self.yawn_count = 0
        self.eye_closed_frames = 0
        self.blink_times = []
        self.yawn_frames = 0
        self.EYE_AR_THRESH = 0.25
        self.MOUTH_AR_THRESH = 0.60
        self.EYE_AR_CONSEC_FRAMES = 3
        self.MOUTH_AR_CONSEC_FRAMES = 15
        self.BLINK_THRESHOLD = 19
        self.SLEEPINESS_THRESHOLD = 5
        self.eye_closed_start_time = None

    def start_capture(self):
        if not self.is_capturing:
            self.cap = cv2.VideoCapture(0)
            self.is_capturing = True
            self.update_frame()

    def stop_capture(self):
        if self.is_capturing:
            self.cap.release()
            self.is_capturing = False
            self.video_label.config(image='')

    def quit_app(self):
        if self.is_capturing:
            self.cap.release()
        self.root.quit()

    def update_frame(self):
        if self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                for face in faces:
                    landmarks = predictor(gray, face)

                    for i in range(68):
                        x = landmarks.part(i).x
                        y = landmarks.part(i).y
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                    left_eye_region = get_eye_region(landmarks, left=True)
                    right_eye_region = get_eye_region(landmarks, left=False)

                    left_centroid = get_centroid(left_eye_region)
                    right_centroid = get_centroid(right_eye_region)

                    left_ear = eye_aspect_ratio(left_eye_region)
                    right_ear = eye_aspect_ratio(right_eye_region)

                    ear = (left_ear + right_ear) / 2.0

                    if ear < self.EYE_AR_THRESH:
                        if self.eye_closed_start_time is None:
                            self.eye_closed_start_time = time.time()
                        self.eye_closed_frames += 1
                    else:
                        if self.eye_closed_frames >= self.EYE_AR_CONSEC_FRAMES:
                            self.blink_count += 1
                            self.blink_times.append(time.time())
                            self.blink_label.config(text=f"Parpadeos: {self.blink_count}")
                            self.check_sleepiness()
                        self.eye_closed_frames = 0
                        self.eye_closed_start_time = None

                    if self.eye_closed_start_time and (time.time() - self.eye_closed_start_time) > self.SLEEPINESS_THRESHOLD:
                        self.sleepy_label.config(text="¡Somnolencia detectada!")
                        self.root.after(3000, self.clear_sleepy_label)

                    if is_distracted(left_centroid, right_centroid, face):
                        self.alert_label.config(text="")
                    else:
                        self.alert_label.config(text="¡Atención perdida!")

                    cv2.polylines(frame, [left_eye_region], True, (0, 255, 0), 2)
                    cv2.polylines(frame, [right_eye_region], True, (0, 255, 0), 2)
                    cv2.circle(frame, left_centroid, 2, (0, 255, 255), -1)
                    cv2.circle(frame, right_centroid, 2, (0, 255, 255), -1)

                    mouth_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
                    mar = mouth_aspect_ratio(mouth_region)

                    if mar > self.MOUTH_AR_THRESH:
                        self.yawn_frames += 1
                    else:
                        if self.yawn_frames >= self.MOUTH_AR_CONSEC_FRAMES:
                            self.yawn_count += 1
                            self.yawn_label.config(text=f"Bostezos: {self.yawn_count}")
                        self.yawn_frames = 0

                    cv2.polylines(frame, [mouth_region], True, (255, 0, 0), 2)

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                self.root.after(10, self.update_frame)

    def check_sleepiness(self):
        if len(self.blink_times) > 0:
            interval = time.time() - self.blink_times[-1]
            if interval < 60:
                self.sleepy_label.config(text="¡Somnolencia detectada!")
                self.root.after(3000, self.clear_sleepy_label)

    def clear_sleepy_label(self):
        self.sleepy_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttentionApp(root)
    root.mainloop()
