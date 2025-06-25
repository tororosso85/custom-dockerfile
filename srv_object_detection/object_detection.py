import cv2
import dlib
import os
from datetime import datetime
import os

# Carica l'URL del flusso MJPEG dalle variabili d'ambiente
MJPEG_STREAM_URL = os.getenv('MJPEG_STREAM_URL', 'about:blank')

# Crea una cartella per le immagini salvate
if not os.path.exists("faces"):
    os.makedirs("faces")

# Inizializza il rilevamento dei volti con dlib
detector = dlib.get_frontal_face_detector()

# Inizializza il descrittore facciale per il riconoscimento
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Modello di punti di riferimento facciali

# Apri il flusso MJPEG
cap = cv2.VideoCapture(MJPEG_STREAM_URL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Errore nel recupero del frame!")
        break

    # Converti l'immagine in scala di grigi (dlib funziona meglio su immagini in bianco e nero)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Rileva i volti nel frame
    faces = detector(gray)

    for face in faces:
        # Estrai i punti di riferimento del volto
        landmarks = sp(gray, face)

        # Crea una finestra di delimitazione attorno al volto
        top, right, bottom, left = (face.top(), face.right(), face.bottom(), face.left())

        # Estrai il volto dall'immagine
        face_image = frame[top:bottom, left:right]

        # Salva il volto
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"faces/{timestamp}.jpg"
        cv2.imwrite(filename, face_image)

        # Disegna un rettangolo attorno al volto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Mostra il frame con i volti riconosciuti
    cv2.imshow('MJPEG Streaming', frame)

    # Premi 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la cattura e chiudi la finestra
cap.release()
cv2.destroyAllWindows()
