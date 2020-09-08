from agent import Agent
from mcts import MCTSPlayer, MCTSPurePlayer
from policy_value_net import PolicyValueNet
import time
import os

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
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    size = 6  # 棋盘大小
    n_in_row = 4  # 几子连线
    model_file =  os.path.join(curr_dir, '../data/save/06_model_%s_%s.pth'%(size,n_in_row))

    try:
        agent = Agent(size=size, n_in_row=n_in_row)
        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        best_policy = PolicyValueNet(size, model_file = model_file)
        mcts_ai_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        mcts_player = MCTSPurePlayer(c_puct=5, n_playout=2000)

        # human player
        human = Human(agent,is_show=1)

        # set start_player=0 for human first
        agent.start_play(human, mcts_ai_player, start_player=0, is_shown=1)
    except KeyboardInterrupt:
        print('quit')


if __name__ == '__main__':
    run()