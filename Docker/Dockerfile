FROM nvidia/cuda:10.2-runtime-ubuntu18.04

COPY get-pip.py get-pip.py

RUN \
    mv /etc/apt/sources.list.d /etc/apt/sources.list.d.bak && \
    apt-get update -y && \
    apt-get install -y python3 python3-pip && \
    update-alternatives --remove-all python && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.6 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    pip config set global.index-url https://pypi.douban.com/simple/ && \
    pip install torch torchvision torchaudio && \
    pip install opencv-python opencv-contrib-python && \
    rm -rf /root/.cache/pip
