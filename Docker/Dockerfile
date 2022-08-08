FROM nvidia/cuda:10.2-runtime-ubuntu18.04

COPY get-pip.py get-pip.py

RUN \
    mv /etc/apt/sources.list.d /etc/apt/sources.list.d.bak && \
    apt-get update -y && \
    apt-get install -y python3 && \
    apt install -y python3-distutils && \
    ln -s /usr/bin/python3.6 /usr/bin/python && \
    python get-pip.py && \
    rm get-pip.py && \
    pip config set global.index-url https://pypi.douban.com/simple/ &&\
    pip install torch torchvision torchaudio && \
    pip install opencv-python opencv-contrib-python && \
    rm -rf /root/.cache/pip
