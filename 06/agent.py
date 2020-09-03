import numpy as np
from game import FiveChessEnv

# 游戏的AI封装
class Agent(object):
    def __init__(self, size, n_in_row):
        self.size = size
        self.n_in_row = n_in_row
        self.env = FiveChessEnv(size=self.size, n_in_row=self.n_in_row)
        self.env.reset()
        self.board = self.env.chessboard

    # 走棋
    def step(self, action):
        self.env.step(action)

    # 位置转action
    def positions_to_actions(self, positions):
        return [(i%self.size, i//self.size) for i in positions]

    # action转位置
    def actions_to_positions(self, actions):
        return [x+y*self.size for x,y in actions]

    # 返回 （是否胜利， 胜利玩家） ，如果没有胜利者，胜利玩家 = -1
    def game_end(self):
        return self.env.check_terminal()

    def get_availables(self):
        return self.env.availables

    # 返回 [1, 4, size, size]
    def current_state(self):
        square_state = np.zeros((4, self.size, self.size))
        curr_player_id = self.env.current_player
        # 前面2层是自己和对手的棋
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y]!=0:
                    idx = 0 if self.board[x][y]==self.env.colors[curr_player_id] else 1
                    square_state[idx][x][y] = 1.0
        # 第三层为最后一步
        if self.env.last_action!=None:
            x,y = self.env.last_action
            square_state[2][x][y] = 1.0
        # 第四层为如果当前用户是先手则为1
        if curr_player_id == 0:
            square_state[3][:,:] = 1.0
        return square_state

    # 使用 mcts 训练，重用搜索树，并保存数据
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.env.reset()
        p1, p2 = self.env.players
        states, mcts_probs, current_players = [], [], []
        while True:
            # temp 权重 ，return_prob 是否返回概率数据
            action, move_probs = player.get_action(self, temp=temp, return_prob=1)
            # store the data
            states.append(self.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.env.current_player)
            # perform a move
            self.step(action)
            if is_shown:
                self.env.render()
            end, winner = self.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)