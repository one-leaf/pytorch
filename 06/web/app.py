from flask import Flask, render_template, jsonify, request
import random
import zmq
import os
import hashlib
import json

curr_dir = os.path.dirname(__file__)

HOST = '192.168.1.10'
PORT = 5555
TASK_SOCKET = zmq.Context().socket(zmq.REQ)
TASK_SOCKET.connect('tcp://{}:{}'.format(HOST, PORT))
print("connected to zmq server")

app = Flask(__name__, static_url_path='')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/step', methods=['POST'])
def step():
    actions = request.json['actions']
    actions_str = json.dumps(actions)
    # 检查是否存在Cache，如果存在，直接返回
    hl = hashlib.md5()
    hl.update(actions_str.encode("UTF-8"))
    md5 = hl.hexdigest()
    filename = os.path.join(curr_dir, "cache", md5[:2], md5)
    if os.path.exists(filename):
        return open(filename,encoding="UTF-8").read()

    # 设置超时时间
    # poll = zmq.Poller()
    # poll.register(TASK_SOCKET, zmq.POLLIN)
    print("send to zmq:",actions_str)
    TASK_SOCKET.send_string(actions_str)
    print("witting from zmq")

    # socks = dict(poll.poll(600000))
    # if TASK_SOCKET in socks and socks.get(TASK_SOCKET) == zmq.POLLIN:
    response = TASK_SOCKET.recv()
    response = response.decode('utf-8')
    print("recv from zmq:", response)
    # 保存到Cache目录
    save_path = os.path.dirname(filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if response.find("error")<0: 
        with open(filename, "w") as f:
            f.write(response)
    # else:
    #     response = "{'action':[]}"

    result=json.loads(response)
    return jsonify(result)

if __name__ =="__main__":
    app.run(debug=True, port=8080, host="0.0.0.0")