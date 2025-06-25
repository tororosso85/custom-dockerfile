import cv2
import numpy as np
import os
import threading
import face_recognition
from openalpr import Alpr
from datetime import datetime
import dlib
import pickle
import requests  # Per l'integrazione con Home Assistant

# Carica l'URL del flusso MJPEG dalle variabili d'ambiente
MJPEG_STREAM_URL = os.getenv('MJPEG_STREAM_URL', 'about:blank')

# Home Assistant API
HOME_ASSISTANT_URL = os.getenv('HOME_ASSISTANT_URL', 'http://localhost:8123')  # Modifica l'URL di Home Assistant
LONG_LIVED_ACCESS_TOKEN = os.getenv('LONG_LIVED_ACCESS_TOKEN', '')  # Token di accesso

if MJPEG_STREAM_URL == 'about:blank' or not LONG_LIVED_ACCESS_TOKEN:
    print("Attenzione: MJPEG_STREAM_URL o access_token non configurati correttamente.")
    exit()

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

# Inizializza il database dei volti conosciuti
known_face_encodings = []
known_face_names = []

# Inizializza il database delle targhe conosciute
known_plates = []

# Funzione per caricare il database dei volti (se esiste)
def carica_database_volti():
    global known_face_encodings, known_face_names
    if os.path.exists("known_faces.pkl"):
        with open("known_faces.pkl", "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)
            print("Database volti caricato con successo!")

# Funzione per caricare il database delle targhe (se esiste)
def carica_database_targhe():
    global known_plates
    if os.path.exists("known_plates.pkl"):
        with open("known_plates.pkl", "rb") as f:
            known_plates = pickle.load(f)
            print("Database targhe caricato con successo!")

# Funzione per salvare il database dei volti
def salva_database_volti():
    with open("known_faces.pkl", "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
        print("Database volti salvato.")

# Funzione per salvare il database delle targhe
def salva_database_targhe():
    with open("known_plates.pkl", "wb") as f:
        pickle.dump(known_plates, f)
        print("Database targhe salvato.")

# Funzione per inviare l'evento a Home Assistant
def invia_evento_home_assistant(nome_volto):
    headers = {
        "Authorization": f"Bearer {LONG_LIVED_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    data = {
        "event": "face_recognized",
        "data": {
            "name": nome_volto,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    url = f"{HOME_ASSISTANT_URL}/api/events/face_recognized"
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            print(f"Evento inviato a Home Assistant: {nome_volto}")
        else:
            print(f"Errore nell'invio dell'evento a Home Assistant: {response.status_code}")
    except Exception as e:
        print(f"Errore nella connessione con Home Assistant: {e}")

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

            # Salva l'immagine del volto
            face_img = frame[top:bottom, left:right]
            salva_immagine(name, face_img, "volti")

        # Invia l'evento a Home Assistant quando un volto è riconosciuto
        invia_evento_home_assistant(name)

# Funzione per il riconoscimento targhe
def rileva_targhe(frame):
    results = alpr.recognize_ndarray(frame)

    for plate in results['plates']:
        targa = plate['plate']

        if targa not in known_plates:
            known_plates.append(targa)

            # Salva l'immagine della targa
            salva_immagine(targa, frame, "targhe")

# Funzione per il salvataggio delle immagini
def salva_immagine(categoria, img, nome_cartella):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{nome_cartella}/{categoria}_{timestamp}.jpg"
    cv2.imwrite(filename, img)
    print(f"Immagine salvata in: {filename}")

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

        # Rilevamento volti
        rileva_volti(frame)

        # Rilevamento targhe
        rileva_targhe(frame)

    cap.release()

# Carica i database iniziali
carica_database_volti()
carica_database_targhe()

# Avvia il flusso video
flusso_thread = threading.Thread(target=cattura_flusso)
flusso_thread.start()

# Attendi la fine del thread principale
flusso_thread.join()

# Salva i database aggiornati
salva_database_volti()
salva_database_targhe()

print("Elaborazione completata.")
