1   days: 1
    reward:
        if state.game.terminal: return -2
        if state.game.state==1 and _emptyCount>=3:
            v = -1
        else:
            v = self.search(state)
    v: (q - q_mean)/q_std
    a: q_t+1 - q_t
    n_playout: 64 
    loss: a + p + v
    out: 
        steps:229
        piececount: 26
        score_mcts: 1
        pacc: 0.70

2   days: 1
    reward:
        if state.game.terminal: return -2
    v: (1 ~ -1)/v_std 
    a: q_t+1 - q_t
    n_playout: 64 
    loss: a + p + v + n
    out: 
        steps:191
        piececount: 22
        score_mcts: 0
        pacc: 0.89

3   days: 1
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: q_t+1 - q_t
    n_playout: 64 
    loss: a + p + v + n
    out: 
        steps: 275
        piececount: 28
        score_mcts: 0.1
        pacc: 0.92

4   days: 1
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: q - v
    n_playout: 64 
    loss: a + p + v + n
    out: 
        steps: 165
        piececount: 24
        score_mcts: 0
        pacc: 0.89

3   days: 2
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: q_t+1 - q_t
    n_playout: 64 
    loss: a + p + v + n
    out: 
        steps: 268
        piececount: 29
        score_mcts: 0.1
        pacc: 0.91


