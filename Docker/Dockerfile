FROM nvidia/cuda:10.2-runtime-ubuntu18.04

RUN rm /etc/apt/sources.list.d/* && \
    sed -i s@/archive.ubuntu.com/@/mirrors.163.com/@g /etc/apt/sources.list && \
    apt update -y && \
    apt install -y software-properties-common curl

RUN DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get -y install tzdata

RUN  apt-add-repository -y ppa:deadsnakes/ppa && \
    apt update -y && \
    apt install -y python3.10 python3.10-distutils && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 10 

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    pip config set global.index-url https://pypi.douban.com/simple/

RUN pip install torch torchvision torchaudio 

RUN pip install Pillow opencv-python opencv-contrib-python 

RUN pip install pygame 

RUN rm -rf /root/.cache/pip
