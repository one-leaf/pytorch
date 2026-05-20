参考： https://www.jianshu.com/p/e5a21db51b57

== 确定母机显卡驱动版本 ==
   sudo nvidia-smi

== 查看支持的cuda版本 ==

   https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

== 查看pytorch是否支持该cuda版本 ==

   https://pytorch.org/


==  选择pip，确定安装方式，创建Dockfile ==

=== pip 安装来源 ===

   https://bootstrap.pypa.io/get-pip.py

== docker pull base image ==
   docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04
   docker tag  swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04  docker.io/nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04

== docker build ==
  docker build -t pytorch-opencv:v12 .