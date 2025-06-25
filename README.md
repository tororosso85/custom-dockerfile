# Docker Build Example

Questo progetto dimostra come creare un workflow GitHub Actions per costruire più container Docker (Python e Node.js) in parallelo.

## Struttura del Progetto

- **srv_ffmpeg**: Una build per ffmpeg
- **srv_object_detection**: Una build per opencv
- **srv_sonarscanner**: Una build per sonarscanner da usare in coppia con sonarqube

Il workflow GitHub Actions costruisce e tagga le immagini Docker senza fare il push su un registry esterno.

## Come Usarlo

1. Modifica il file '.github/workflows/build-all-containers.yml', aggiungendo il nome della cartella in cui caricare il Dockerfile
2. Fai un push al branch `main`.
3. Il workflow costruirà automaticamente i container.
