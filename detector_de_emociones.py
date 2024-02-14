import cv2
from keras.models import load_model
import numpy as np

# Carga el clasificador preentrenado para la detección de caras
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

# Carga el modelo preentrenado para la clasificación de expresiones
emotion_model = load_model('emotion_model.hdf5', compile=False)

# Define las etiquetas de las expresiones
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Inicia la captura de video
video_capture = cv2.VideoCapture(0)

while True:
    # Captura el fotograma por fotograma
    ret, frame = video_capture.read()

    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta caras en la imagen
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Para cada cara detectada
    for (x, y, w, h) in faces:
        # Extrae la región de interés (ROI) que es la cara
        roi_gray = gray[y:y+h, x:x+w]

        # Cambia el tamaño de la imagen para adaptarse al modelo de emociones
        roi_gray_resized = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        # Normaliza la imagen
        roi_gray_resized = roi_gray_resized / 255.0

        # Agrega una dimensión adicional para la predicción del modelo
        roi_gray_resized = np.expand_dims(roi_gray_resized, axis=0)
        roi_gray_resized = np.expand_dims(roi_gray_resized, axis=-1)

        # Realiza la predicción de la emoción
        emotion_prediction = emotion_model.predict(roi_gray_resized)
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]


        # Dibuja un rectángulo alrededor de la cara y muestra la emoción
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Muestra el fotograma resultante
    cv2.imshow('Video', frame)

    # Rompe el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos cuando el script termina
video_capture.release()
cv2.destroyAllWindows()
