# Usa una base image di Debian per ARM (compatibile con Raspberry Pi 4)
FROM python:slim

# Imposta la variabile di ambiente per non chiedere conferme durante l'installazione dei pacchetti
ENV DEBIAN_FRONTEND=noninteractive

# Imposta PYTHONUNBUFFERED per evitare buffering dell'output
ENV PYTHONUNBUFFERED 1

#Crea le cartelle
RUN { mkdir -p /app && mkdir -p /app/faces; } > /dev/null

# Aggiorna il sistema e installa le dipendenze di sistema
RUN { apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    python3-dev \
    python3-pip \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    libcurl4-openssl-dev \
    liblog4cpp5-dev \
    libopencv-dev \
    libjpeg-dev \
    zlib1g-dev \
    git \
    wget \
    bzip \
    libssl-dev && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/tmp/* && \
    rm -Rf /usr/share/doc && \
    rm -Rf /usr/share/man; } > /dev/null

# Clonazione del repository OpenALPR, compila e installa
RUN { git clone https://github.com/openalpr/openalpr.git /openalpr && \
    cd /openalpr && \
    make && \
    sudo make install && \
    cp /etc/openalpr/openalpr.conf /etc/openalpr/openalpr.conf.default; } > /dev/null

#Per il riconoscimento del viso
RUN { cd /app && wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
	bunzip2 *.bz2  && \
	rm *.bz2; } > /dev/null

# Installa tutte le dipendenze Python in un unico comando pip3
RUN { pip3 install --no-cache-dir opencv-python-headless face_recognition openalpr numpy dlib; } > /dev/null

# Crea la cartella del progetto
WORKDIR /app

# Copia i file del progetto nel container
# COPY *.py /app

# Comando per eseguire il programma
CMD ["python3", "object_detection.py"]
