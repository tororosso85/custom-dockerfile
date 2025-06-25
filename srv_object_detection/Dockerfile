# Utilizza un'immagine base con Python e OpenCV
FROM python:slim

ENV DEBIAN_FRONTEND=noninteractive

# Imposta PYTHONUNBUFFERED per evitare buffering dell'output
ENV PYTHONUNBUFFERED 1

RUN { mkdir -p /app && mkdir -p /app/faces; } > /dev/null

# Installa le dipendenze di sistema
RUN { apt-get update -y && apt-get install -y --no-install-recommends build-essential make cmake g++ gcc libffi-dev python3-dev libboost-all-dev wget bzip2 \
    libopenblas-dev libopencv-dev libjpeg-dev libpython3-dev libatlas-base-dev libhdf5-dev libprotobuf-dev protobuf-compiler && \
	rm -rf /var/lib/apt/lists/* && \
	rm -rf /var/tmp/* && \
	rm -Rf /usr/share/doc && \
	rm -Rf /usr/share/man; } > /dev/null

# Installa le librerie Python necessarie
RUN { pip install --no-cache-dir opencv-python-headless dlib; } > /dev/null

RUN { cd /app && wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
	bunzip2 *.bz2  && \
	rm *.bz2; } > /dev/null

# Copia lo script Python nel container
COPY object_detection.py /app/object_detection.py

# Imposta la directory di lavoro
WORKDIR /app

# Comando per avviare lo script Python
ENTRYPOINT ["python", "object_detection.py"]
