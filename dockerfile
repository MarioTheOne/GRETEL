FROM tensorflow/tensorflow:latest-gpu

ARG USERNAME=coder
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -p "$(openssl passwd -1 $USERNAME)"

# Setup VS code compatibility for easy interaction with code inside container
RUN mkdir -p /home/$USERNAME/.vscode-server/extensions \
        /home/$USERNAME/.vscode-server-insiders/extensions \
    && chown -R $USERNAME \
        /home/$USERNAME/.vscode-server \
        /home/$USERNAME/.vscode-server-insiders

RUN apt update \
 && apt install -y \
    curl \
    locales \
    nano \
    ssh \
    sudo \
    bash \
    git \
    make \
    gcc \
    wget\
    build-essential \
    python3-dev \
    python3-tk

RUN mkdir -p /home/$USERNAME/.gretel/data && chown $USERNAME:$USERNAME /home/$USERNAME/.gretel
VOLUME /home/$USERNAME/.gretel
COPY ./ /home/$USERNAME/gretel

# Install project requirements
COPY ./requirements.txt /home/$USERNAME/requirements.txt
RUN python3 -m pip install -r /home/$USERNAME/requirements.txt
RUN python3 -m pip install poetry
RUN python3 -m pip install IPython

CMD ["/bin/bash"]