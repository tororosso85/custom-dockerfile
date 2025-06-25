FROM debian:stable-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN { apt-get update && \
        apt-get upgrade -y && \
	apt-get install -y --no-install-recommends wget zip unzip nodejs npm && \
	apt-get clean all && \
	apt-get autoclean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
	rm -rf /var/tmp/* && \
	rm -Rf /usr/share/doc && \
	rm -Rf /usr/share/man; } > /dev/null

RUN { wget https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-6.2.1.4610-linux-aarch64.zip && \
	unzip sonar-scanner*.zip && \
	rm sonar-scanner*.zip && \
	mv sonar-scanner* /usr/local/sonar-scanner; } > /dev/null

ENTRYPOINT ["tail", "-f", "/dev/null"]
