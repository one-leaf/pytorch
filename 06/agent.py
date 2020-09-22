import random
import numpy as np
from game import FiveChess, FiveChessEnv
from itertools import count

# 游戏的AI封装
class Agent(object):
    def __init__(self, size, n_in_row, is_shown=1):
        self.size = size
        self.n_in_row = n_in_row
        self.game = FiveChess(size=self.size, n_in_row=self.n_in_row)
        self.is_shown = is_shown
        if is_shown:
            self.env = FiveChessEnv(self.game)
        self.game.reset()

    # 使用 mcts 训练，重用搜索树，并保存数据
    def start_self_play(self, player, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.game.reset()
        p1, p2 = self.game.players
        states, mcts_probs, current_players = [], [], []
        for i in count():
            # temp 权重 ，return_prob 是否返回概率数据
            action, move_probs = player.get_action(self.game, temp=temp, return_prob=1)
            # store the data
            states.append(self.game.current_state())
            # print(action)
            # print(move_probs.reshape(self.size,self.size))
            # print(states[-1])
            mcts_probs.append(move_probs)
            current_players.append(self.game.current_player)
            # perform a move
            self.game.step(action)
            if self.is_shown:
                self.env.render()
            end, winner = self.game.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if self.is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

    # AI和蒙特卡罗对战
    def start_play(self, player1, player2, start_player=0):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        p1, p2 = self.game.players
        self.game.reset(start_player)
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if self.is_shown:
            self.env.render()
        while True:
            player_in_turn = players[self.game.current_player]
            action = player_in_turn.get_action(self.game)
            self.game.step(action)

            if self.is_shown:
                self.env.render()
            end, winner = self.game.game_end()
            if end:
                if self.is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner