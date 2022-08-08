FROM nvidia/cuda:10.2-runtime-ubuntu18.04

RUN \
    apt update -y && \
    apt install software-properties-common -y && \
    apt-add-repository -y ppa:deadsnakes/ppa && \
    apt update -y && \
    apt install python3.10 python3.10-pip && \
    update-alternatives --remove-all python && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    pip config set global.index-url https://pypi.douban.com/simple/ && \
    pip install torch torchvision torchaudio && \
    pip install opencv-python opencv-contrib-python && \
    pip install Pillow && \
    rm -rf /root/.cache/pip
