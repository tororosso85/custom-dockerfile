import cv2
import numpy as np
import os
import threading
import face_recognition
from openalpr import Alpr
from datetime import datetime
import dlib

# Carica l'URL del flusso MJPEG dalle variabili d'ambiente
MJPEG_STREAM_URL = os.getenv('MJPEG_STREAM_URL', 'about:blank')

if MJPEG_STREAM_URL == 'about:blank':
    print("Attenzione: MJPEG_STREAM_URL non è stato impostato correttamente, utilizzando il valore di default.")
else:
    print(f"URL del flusso MJPEG: {MJPEG_STREAM_URL}")

# Inizializza OpenALPR per il rilevamento targhe
alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "/etc/openalpr/runtime_data")

if not alpr.is_ready():
    print("OpenALPR non è pronto.")
    exit()

# Inizializza il rilevamento dei volti con dlib
detector = dlib.get_frontal_face_detector()

# Inizializza il descrittore facciale per il riconoscimento
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Modello di punti di riferimento facciali

# Crea le cartelle per il salvataggio delle immagini
os.makedirs("volti", exist_ok=True)  # Cartella principale per tutti i volti
os.makedirs("targhe", exist_ok=True)  # Cartella principale per tutte le targhe
os.makedirs("faces", exist_ok=True)  # Cartella per volti

# Database di volti conosciuti
known_face_encodings = []
known_face_names = []

# Funzione per il salvataggio delle immagini
def salva_immagine(categoria, img, nome_cartella):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{nome_cartella}/{categoria}_{timestamp}.jpg"
    cv2.imwrite(filename, img)
    print(f"Immagine salvata in: {filename}")

# Funzione per il riconoscimento volti
def rileva_volti(frame):
    # Converte il frame in RGB per face_recognition
    rgb_frame = frame[:, :, ::-1]  # Converte da BGR a RGB

    # Trova i volti nel frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Confronta il volto con quelli già conosciuti
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Sconosciuto"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            # Aggiungi il volto al database di volti conosciuti
            known_face_encodings.append(face_encoding)
            name = f"persona_{len(known_face_encodings)}"
            known_face_names.append(name)

        # Ritaglia l'immagine del volto
        face_img = frame[top:bottom, left:right]

        # Salva il volto nella cartella appropriata
        salva_immagine(name, face_img, "volti")

# Funzione per il riconoscimento targhe
def rileva_targhe(frame):
    results = alpr.recognize_ndarray(frame)

    for plate in results['plates']:
        targa = plate['plate']

        # Crea la cartella per la targa, se non esiste
        targa_folder = f"targhe/{targa}"
        os.makedirs(targa_folder, exist_ok=True)

        # Salva l'immagine della targa nella cartella corrispondente
        salva_immagine(targa, frame, targa_folder)

# Funzione per il rilevamento movimento e gestione frame
def processa_frame(frame):
    # Rilevamento volti
    rileva_volti(frame)

    # Rilevamento targhe
    rileva_targhe(frame)

# Funzione per la cattura dei frame MJPEG
def cattura_flusso():
    cap = cv2.VideoCapture(MJPEG_STREAM_URL)

    if not cap.isOpened():
        print(f"Impossibile aprire il flusso MJPEG da: {MJPEG_STREAM_URL}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Passa il frame ai thread per il processing
        threading.Thread(target=processa_frame, args=(frame,)).start()

# Avvia il flusso di cattura in un thread separato
flusso_thread = threading.Thread(target=cattura_flusso)
flusso_thread.start()

# Attendi la fine del thread principale
flusso_thread.join()

cap.release()
alpr.unload()
cv2.destroyAllWindows()
