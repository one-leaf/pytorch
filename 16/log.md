1 days:
    reward:
        if state.game.terminal: return -2
        if state.game.state==1 and _emptyCount>=3:
            v = -1
        else:
            v = self.search(state) 
    n_playout: 64 
    loss: a + p + v
    out: 
        steps:229
        piececount: 26
        score_mcts: 1
