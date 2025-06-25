FROM alpine:latest

RUN { apk update && \
	apk add --no-cache ffmpeg && \
	apk cache clean && \
	rm -rf /var/cache/apk/*; } > /dev/null

ENTRYPOINT ["tail", "-f", "/dev/null"]
