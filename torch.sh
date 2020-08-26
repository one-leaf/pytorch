docker run --name torch --gpus device=1 -it -p 5901:5901 -p 6901:6901 -v $PWD:/mnt -w /mnt -v /etc/localtime:/etc/localtime:ro pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime bash


docker run --gpus device=0 -it --rm -v $PWD:/mnt -w /mnt -v /etc/localtime:/etc/localtime:ro oneleaf/pytorch:1.5.1-game ./run.sh