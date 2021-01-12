from agent import Agent
from mcts import MCTSPlayer, MCTSPurePlayer
from model import PolicyValueNet
import time
import os
import random

mouse_click_point = None

def on_mouse_press(x, y, button, modifiers):
    global mouse_click_point
    mouse_click_point = (x, y)       

class Human(object):
    """
    human player
    """
    def __init__(self, agent, is_show=1):
        self.player = None        
        self.agent = agent      
        self.user_click_point=None
        def on_mouse_press(x, y, button, modifiers):
            self.user_click_point = (x, y)
        if is_show:
            self.agent.env.render()
            self.agent.env.viewer.window.on_mouse_press = on_mouse_press

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, state):   
        while True:
            if self.user_click_point!=None:
                action = self.agent.env.point_to_action(self.user_click_point)
                if self.agent.game.is_valid_set_coord(action):
                    self.user_click_point = None
                    break
            self.agent.env.render()
            time.sleep(0.1)
        print(action)
        return action

    def __str__(self):
        return "Human {}".format(self.player)

def run():
    size = 15  # 棋盘大小
    n_in_row = 5  # 几子连线

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(curr_dir, './model/')
    model_file =  os.path.join(model_dir, 'model_%s_%s.pth'%(size,n_in_row))

    try:
        agent = Agent(size=size, n_in_row=n_in_row)
        # ############### human VS AI ###################

        # 神经网络的价值策略
        net_policy = PolicyValueNet(size, model_file = model_file)
        mcts_ai_player = MCTSPlayer(net_policy.policy_value_fn, c_puct=3, n_playout=500, is_selfplay=1)

        # 纯MCTS玩家
        # mcts_player = MCTSPurePlayer(c_puct=5, n_playout=2000)

        # 人类玩家
        human = Human(agent,is_show=1)

        # 设置 start_player=0 AI先走棋
        agent.start_play(mcts_ai_player, human, start_player=0)
        # agent.start_play(human, human, start_player=0 if random.random()>0.5 else 1)
    except KeyboardInterrupt:
        print('quit')


if __name__ == '__main__':
    run()