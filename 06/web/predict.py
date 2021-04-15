import zmq
import json
from datetime import datetime
import traceback
import os

import sys 

curr_dir = os.path.dirname(os.path.abspath(__file__))
p_dir = os.path.join(curr_dir,"../")
sys.path.append(p_dir)

from model import PolicyValueNet  
from mcts import  MCTSPlayer
from game import FiveChess

size = 15
n_in_row = 5
c_puct = 4
n_playout = 300

def main(debug=False):
    
    model_file = os.path.join(curr_dir, "../model/best_model_15_5.pth")
    policy_value_net = PolicyValueNet(size, model_file=model_file)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("Server start on 5555 port")
    while True:
        message = socket.recv()
        try:
            message = message.decode('utf-8')
            actions = json.loads(message)
            print("Received: %s" % message)

            start = datetime.now()
            mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=c_puct, n_playout=n_playout, is_selfplay=0)
            # result = predict
            game = FiveChess(size=size, n_in_row=n_in_row)
            for act in actions:
                step=(act[0],act[1])
                game.step_nocheck(step)

            action, value = mcts_player.get_action(game, return_value=1)

            result = {"action":action, "value": value}

            print(result)

            print('time used: {} sec'.format((datetime.now() - start).total_seconds()))
            socket.send_string(json.dumps(result, ensure_ascii=False))
        except Exception as e:
            traceback.print_exc()
            socket.send_string(json.dumps({"error":str(e)}, ensure_ascii=False))


if __name__ =="__main__":
    main(debug=True)