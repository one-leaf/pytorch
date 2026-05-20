FROM docker.io/nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends software-properties-common curl && \
    rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get -y install tzdata

RUN  apt-add-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends python3.13 python3.13-venv && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.13 10

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    pip config set global.index-url https://pypi.douban.com/simple/
    
RUN pip install torch torchvision torchaudio 

RUN pip install Pillow opencv-python opencv-contrib-python 

RUN pip install pygame 

RUN rm -rf /root/.cache/pip
