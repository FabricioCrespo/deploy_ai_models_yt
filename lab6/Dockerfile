# Usa la imagen oficial de Jenkins como base
FROM jenkins/jenkins:latest

# Instala dependencias necesarias para Anaconda
USER root
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Descarga e instala Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh -O /tmp/anaconda.sh \
    && bash /tmp/anaconda.sh -b -p /opt/anaconda \
    && rm /tmp/anaconda.sh

# Agrega Anaconda al PATH
ENV PATH="/opt/anaconda/bin:$PATH"

# Configura Jenkins para usar Anaconda
USER jenkins