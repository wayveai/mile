FROM carlasim/carla:0.9.11
ARG VNC_PORT=8080
ARG JUPYTER_PORT=8894

USER root
RUN echo 'root:1234' | chpasswd

ENV DEBIAN_FRONTEND noninteractive
ENV APT_INSTALL "apt-get install -y --no-install-recommends"

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

# ==================================================================
# tools
# ------------------------------------------------------------------
RUN $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        zlib1g-dev \
        libjpeg8-dev \
        freeglut3-dev \
        iputils-ping

RUN $APT_INSTALL \
    cmake  \
    protobuf-compiler

# ==================================================================
# SSH
# ------------------------------------------------------------------
RUN apt-get update && $APT_INSTALL openssh-server
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# ==================================================================
# GUI
# ------------------------------------------------------------------
RUN $APT_INSTALL libsm6 libxext6 libxrender-dev mesa-utils

# Setup demo environment variables
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=C.UTF-8 \
    DISPLAY=:0.0 \
    DISPLAY_WIDTH=1024 \
    DISPLAY_HEIGHT=768

RUN apt-get update; \
    $APT_INSTALL \
      fluxbox \
      net-tools \
      novnc \
      supervisor \
      x11vnc \
      xterm \
      xvfb \
      python3-tk \
      libgtk2.0-dev

COPY setup/dep/vnc /vnc
EXPOSE $VNC_PORT

## ==================================================================
## More tools for Carla
## ------------------------------------------------------------------
RUN apt-get update && $APT_INSTALL mesa-vulkan-drivers vulkan-utils xdg-user-dirs xdg-utils
# Make flyby window work with touchpad
RUN sed -i 's/bUseMouseForTouch=False/bUseMouseForTouch=True/' "/home/carla/CarlaUE4/Config/DefaultInput.ini"

## ==================================================================
## Startup
## ------------------------------------------------------------------
COPY setup/on_docker_start.sh /on_docker_start.sh
RUN chmod +x /on_docker_start.sh

# RUN echo 'su - carla -c "./CarlaUE4.sh -RenderOffScreen -nosound -opengl"' > /home/carla/mycarla.sh
# Unreal command line commands: https://docs.unrealengine.com/5.1/en-US/command-line-arguments-in-unreal-engine/
RUN echo "./CarlaUE4.sh -nosound -carla-server -benchmark -fps=10 -quality-level=Epic -RenderOffScreen" > /home/carla/no_screen.sh
RUN chmod +x /home/carla/no_screen.sh

RUN echo "./CarlaUE4.sh -nosound -carla-server -benchmark -fps=10 -quality-level=Epic" > /home/carla/mycarla.sh
RUN chmod +x /home/carla/mycarla.sh

# make bash a default shell for carla user
RUN usermod -s /bin/bash carla

## ==================================================================
## Startup
## ------------------------------------------------------------------
# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p $CONDA_DIR
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

USER carla
WORKDIR /home/carla

COPY environment.yml /home/carla/environment.yml

RUN conda env create -f /home/carla/environment.yml && \
    conda clean --all --yes

RUN conda --version
RUN pwd

USER root
ENTRYPOINT ["/on_docker_start.sh"]
