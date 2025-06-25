# Usa una base image di Debian per ARM (compatibile con Raspberry Pi 4)
FROM python:slim

# Imposta la variabile di ambiente per non chiedere conferme durante l'installazione dei pacchetti
ENV DEBIAN_FRONTEND=noninteractive

# Imposta PYTHONUNBUFFERED per evitare buffering dell'output
ENV PYTHONUNBUFFERED 1

#Crea le cartelle
RUN { mkdir -p /app /app/faces /openalpr; } > /dev/null

# Aggiorna il sistema e installa le dipendenze di sistema
RUN { apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    beanstalkd \
    build-essential \
    bzip2 \
    cmake \
    g++ \
    gcc \
    gfortran \
    git \
    git-core \
    libatlas-base-dev \
    libboost-all-dev \
    libcurl3-dev \
    libcurl4-openssl-dev \
    libffi-dev \
    libgtk-3-dev \
    libhdf5-dev \
    libjpeg-dev \
    liblapack-dev \
    libleptonica-dev \
    liblog4cplus-dev \
    liblog4cpp5-dev \
    libopenblas-dev \
    libopencv-dev \
    libprotobuf-dev \
    libpython3-dev \
    libssl-dev \
    libtesseract-dev \
    libtiff-dev \
    libtool \
    libx11-dev \
    make \
    protobuf-compiler \
    python3-dev \
    tesseract-ocr \
    wget \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/tmp/* && \
    rm -Rf /usr/share/doc && \
    rm -Rf /usr/share/man; }  > /dev/null

# Installa Tesseract manualmente
RUN { git clone https://github.com/tesseract-ocr/tesseract.git /tesseract && \
    cd /tesseract && \
    git checkout 5.3.0 && \
    ./autogen.sh && \
    ./configure --prefix=/usr --with-extra-includes=/usr/include --with-extra-libraries=/usr/lib && \
    make && \
    make install && \
    ldconfig && \
    cp /usr/local/etc/tesseract/tesseract.conf /usr/etc/tesseract.conf; }

# Clonazione del repository OpenALPR, compila e installa
RUN { git clone https://github.com/openalpr/openalpr.git /openalpr && \
    cd /openalpr/src && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_INSTALL_SYSCONFDIR:PATH=/etc .. && \
    make && \
    make install && \
    cp /etc/openalpr/openalpr.conf /etc/openalpr/openalpr.conf.default; }

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
