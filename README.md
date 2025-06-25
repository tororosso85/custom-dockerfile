# Docker Build Example

Questo progetto dimostra come creare un workflow GitHub Actions per costruire più container Docker (Python e Node.js) in parallelo.

## Struttura del Progetto

- **srv_sonarscanner**: Una build per sonarscanner da usare in coppia con sonarqube

Il workflow GitHub Actions costruisce e tagga le immagini Docker senza fare il push su un registry esterno.

## Come Usarlo

1. Fai un push al branch `main`.
2. Il workflow costruirà automaticamente i container per Python e Node.js.
