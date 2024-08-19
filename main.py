import cv2  # Importa OpenCV para procesamiento de imágenes
import dlib  # Importa dlib para detección y predicción de características faciales
import numpy as np  # Importa numpy para operaciones matemáticas
import tkinter as tk  # Importa tkinter para crear una interfaz gráfica
from PIL import Image, ImageTk  # Importa PIL para manipulación de imágenes
from scipy.spatial import distance as dist  # Importa scipy para calcular distancias
import time  # Importa time para medir el tiempo

# Inicializar detector de rostros y predictor de rasgos faciales
detector = dlib.get_frontal_face_detector()  # Carga el detector de rostros frontal de dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Carga el predictor de 68 puntos faciales de dlib

# Función para obtener la región de los ojos
def get_eye_region(landmarks, left=True):
    if left:
        points = [36, 37, 38, 39, 40, 41]  # Índices de los puntos del ojo izquierdo
    else:
        points = [42, 43, 44, 45, 46, 47]  # Índices de los puntos del ojo derecho
    # Extrae las coordenadas de los puntos del ojo
    region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
    return region

# Función para calcular el centroide de una región
def get_centroid(region):
    M = cv2.moments(region)  # Calcula los momentos de la región
    if M["m00"] == 0:
        return 0, 0  # Evita división por cero
    cX = int(M["m10"] / M["m00"])  # Calcula la coordenada X del centroide
    cY = int(M["m01"] / M["m00"])  # Calcula la coordenada Y del centroide
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
    A = dist.euclidean(mouth[3], mouth[9])  # 63, 67
    B = dist.euclidean(mouth[2], mouth[10])  # 62, 66
    C = dist.euclidean(mouth[0], mouth[6])  # 60, 64
    mar = (A + B) / (2.0 * C)
    return mar

# Función para determinar si hay pérdida de atención
def is_distracted(left_centroid, right_centroid, face_rect):
    # Calcula el centro del rostro
    face_center_x = (face_rect.left() + face_rect.right()) // 2
    face_center_y = (face_rect.top() + face_rect.bottom()) // 2
    threshold = 0.26  # Umbral para determinar distracción
    # Calcula la distancia entre el centroide del ojo y el centro del rostro
    left_dist = np.linalg.norm(np.array(left_centroid) - np.array([face_center_x, face_center_y]))
    right_dist = np.linalg.norm(np.array(right_centroid) - np.array([face_center_x, face_center_y]))
    # Si la distancia excede el umbral, se considera distracción
    if left_dist > face_rect.width() * threshold or right_dist > face_rect.width() * threshold:
        return True
    return False

# Interfaz gráfica con tkinter
class AttentionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Pérdida de Atención")
        self.video_label = tk.Label(root)
        self.video_label.pack()
        self.alert_label = tk.Label(root, text="", font=("Arial", 24), fg="red")
        self.alert_label.pack()
        self.blink_label = tk.Label(root, text="Parpadeos: 0", font=("Arial", 24), fg="blue")
        self.blink_label.pack()
        self.yawn_label = tk.Label(root, text="Bostezos: 0", font=("Arial", 24), fg="purple")
        self.yawn_label.pack()
        self.sleepy_label = tk.Label(root, text="", font=("Arial", 24), fg="orange")
        self.sleepy_label.pack()
        self.cap = cv2.VideoCapture(0)  # Captura de video desde la cámara
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
        self.SLEEPINESS_THRESHOLD = 5  # Threshold for sleepiness in seconds
        self.eye_closed_start_time = None
        self.update_frame()  # Actualiza el marco de video

    def update_frame(self):
        ret, frame = self.cap.read()  # Lee un marco de la cámara
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte el marco a escala de grises
            faces = detector(gray)  # Detecta rostros en el marco

            for face in faces:
                landmarks = predictor(gray, face)  # Predice los puntos faciales en el rostro detectado
                
                # Dibuja todos los puntos faciales
                for i in range(68):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                left_eye_region = get_eye_region(landmarks, left=True)  # Obtiene la región del ojo izquierdo
                right_eye_region = get_eye_region(landmarks, left=False)  # Obtiene la región del ojo derecho

                left_centroid = get_centroid(left_eye_region)  # Calcula el centroide del ojo izquierdo
                right_centroid = get_centroid(right_eye_region)  # Calcula el centroide del ojo derecho

                left_ear = eye_aspect_ratio(left_eye_region)  # Calcula el EAR del ojo izquierdo
                right_ear = eye_aspect_ratio(right_eye_region)  # Calcula el EAR del ojo derecho

                ear = (left_ear + right_ear) / 2.0

                # Verifica si los ojos están cerrados
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

                # Verifica si los ojos han estado cerrados más de 5 segundos
                if self.eye_closed_start_time and (time.time() - self.eye_closed_start_time) > self.SLEEPINESS_THRESHOLD:
                    self.sleepy_label.config(text="¡Somnolencia detectada!")
                    # Llama a la función para borrar el texto de somnolencia detectada después de 3 segundos
                    self.root.after(3000, self.clear_sleepy_label)
                #else:
                    #self.sleepy_label.config(text="")

                # Verifica si hay distracción
                if is_distracted(left_centroid, right_centroid, face):
                    self.alert_label.config(text="")
                else:
                    self.alert_label.config(text="¡Atención perdida!")

                # Dibuja las regiones de los ojos y los centroides
                cv2.polylines(frame, [left_eye_region], True, (0, 255, 0), 2)
                cv2.polylines(frame, [right_eye_region], True, (0, 255, 0), 2)
                cv2.circle(frame, left_centroid, 2, (0, 255, 255), -1)
                cv2.circle(frame, right_centroid, 2, (0, 255, 255), -1)

                # Procesamiento de la boca para detección de bostezos
                mouth_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
                mar = mouth_aspect_ratio(mouth_region)

                # Incrementa el contador de bostezos si MAR supera el umbral
                if mar > self.MOUTH_AR_THRESH:
                    self.yawn_frames += 1
                else:
                    if self.yawn_frames >= self.MOUTH_AR_CONSEC_FRAMES:
                        self.yawn_count += 1
                        self.yawn_label.config(text=f"Bostezos: {self.yawn_count}")
                    self.yawn_frames = 0

                cv2.polylines(frame, [mouth_region], True, (255, 0, 0), 2)  # Dibuja la región de la boca

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        self.video_label.after(10, self.update_frame)

    def check_sleepiness(self):
        # Elimina los parpadeos que ocurrieron hace más de 60 segundos
        current_time = time.time()
        self.blink_times = [t for t in self.blink_times if current_time - t <= 60]
        blink_rate = len(self.blink_times) / 60.0
        if blink_rate < self.BLINK_THRESHOLD:
            self.sleepy_label.config(text="¡Somnolencia detectada!")
            # Llama a la función para borrar el texto de somnolencia detectada después de 3 segundos
            self.root.after(3000, self.clear_sleepy_label)

    def clear_sleepy_label(self):
        self.sleepy_label.config(text="")

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttentionApp(root)
    root.mainloop()
