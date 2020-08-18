# pytorch
PyTorch 测试

== 安装 ==

 python3 -m pip install torch torchvision


== 参考来源 ==

 http://pytorch123.com/

== 其他 ==

xvfb-run --listen-tcp --server-num 1 --auth-file /tmp/xvfb.auth -s "-screen 0 1400x900x24" python 04/14.py

x11vnc -rfbport 5901 -rfbauth /root/.vnc/passwd -display :1 -forever -auth /tmp/xvfb.auth