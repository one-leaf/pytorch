import random
import numpy as np
from game import FiveChess, FiveChessEnv
from itertools import count
from mcts import MCTSPlayer, MCTSPurePlayer
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
    def start_self_play(self, player1, player2=None, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.game.reset(start_player=0 if random.random()>0.5 else 1, need_shuffle_availables = player2 is None)
        p1, p2 = self.game.players

        if not player2 is None:
            player1.set_player_ind(p1)
            player2.set_player_ind(p2)             
            players = {p1: player1, p2: player2}              
        else:
            players = {p1: player1, p2: player1}

        states, mcts_probs, current_players = [], [], []
        for i in count():
            # temp 权重 ，return_prob 是否返回概率数据
            player_in_turn = players[self.game.current_player]
            if isinstance(player_in_turn, MCTSPlayer):
                action, move_probs = player_in_turn.get_action(self.game, temp=temp, return_prob=1)
                # action, move_probs = player_in_turn.get_action(self.game, temp=temp, return_prob=1)
                # store the data
                # states.append(self.game.current_state())
                # print(move_probs.reshape(self.size,self.size))
                # print(states[-1])
                # mcts_probs.append(move_probs)
                # current_players.append(self.game.current_player)   
                # 如果包含了第二个玩家是MCTS，则AI每一步都需要重置搜索树
                # if not player2 is None:
                player_in_turn.mcts.update_root_with_action(None)             
            else:
                action, move_probs = player_in_turn.get_action(self.game, return_prob=1)
 
            states.append(self.game.current_state())
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
                player1.reset_player()
                if not player2 is None:
                    if winner != -1:
                        winner_play= player1 if winner == player1.player else player2
                        print("Game end. Winner is player:", winner_play)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

    # AI和蒙特卡罗对战
    def start_play(self, player1, player2, start_player=0, need_shuffle_availables=False):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        p1, p2 = self.game.players
        self.game.reset(start_player=start_player, need_shuffle_availables = need_shuffle_availables)

        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if self.is_shown:
            self.env.render()
        while True:
            player_in_turn = players[self.game.current_player]
            action = player_in_turn.get_action(self.game)
            self.game.step(action)

            if self.game.check_will_win():
                print(action,"will win!")

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

    def start_self_evaluate(self, player1, player2, temp=0.1, start_player=0):
        """ 两个mcts 自我评测
        """
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

        states, mcts_probs, current_players = [], [], []
        for i in count():
            player_in_turn = players[self.game.current_player]
            action, move_probs = player_in_turn.get_action(self.game, temp=temp, return_prob=1)
            states.append(self.game.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.game.current_player)  

            self.game.step(action)

            if self.is_shown:
                self.env.render()
            end, winner = self.game.game_end()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player1.reset_player()
                player2.reset_player()

                return winner, zip(states, mcts_probs, winners_z)
