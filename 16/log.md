1 days:
    reward:
        if state.game.terminal: return -2
        if state.game.state==1 and _emptyCount>=3:
            v = -1
        else:
            v = self.search(state)
    v: q
    a: q_t+1 - q_t
    n_playout: 64 
    loss: a + p + v
    out: 
        steps:229
        piececount: 26
        score_mcts: 1
        pacc: 0.70

1 days:
    reward:
        if state.game.terminal: return -2
    v: (1 ~ -1)/std 
    a: q_t+1 - q_t
    n_playout: 64 
    loss: a + p + v + n
    out: 
        steps:191
        piececount: 22
        score_mcts: 0
        pacc: 0.89
