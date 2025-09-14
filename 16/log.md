1   days: 1
    reward:
        if state.game.terminal: return -2
        if state.game.state==1 and _emptyCount>=3:
            v = -1
        else:
            v = self.search(state)
    v: (q - q_mean)/q_std
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_playout: 64 
    model: a = mcts_probs - log_probs 
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
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_playout: 64 
    model: a = mcts_probs - log_probs 
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
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_playout: 64 
    model: a = mcts_probs - log_probs 
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
    a: ((q - v) - a_mean)/a_std
    n_playout: 64 
    model: a = mcts_probs - log_probs 
    loss: a + p + v + n
    out: 
        steps: 165
        piececount: 24
        score_mcts: 0
        pacc: 0.89

4.1  days: 2
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_playout: 64 
    model: a = mcts_probs - log_probs 
    loss: a + p + v + n
    out: 
        steps: 268
        piececount: 29
        score_mcts: 0.1
        pacc: 0.91

5   days: 1
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: ((q - v) - a_mean)/a_std
    n_playout: 64 
    model: a = log_probs - mcts_probs
    loss: a + p + v + n
    out: 
        steps: 
        piececount: 
        score_mcts: 
        pacc: 

6   days: 1
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_playout: 64 
    model: a = mcts_probs - log_probs 
    loss: a + p + v + n
    out: 
        steps: 214
        piececount: 23
        score_mcts: 0.08
        pacc: 0.93

7   days: 1
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_playout: 128
    model: a = mcts_probs - log_probs 
    loss: a + p + v + n
    out: 
        steps: 169
        piececount: 19
        score_mcts: 0.03
        pacc: 0.91

8  days: 1
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_search_count: 64
    model: a = mcts_probs - log_probs 
    loss: a + p + v + n
    out: 
        steps: 166
        piececount: 24
        score_mcts: 0.02
        pacc: 0.87

9  days: 1
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: ((q_t+1 - q_t) - a_mean)/a_std
    model: a = mcts_probs - log_probs 
    n_search_count: 128
    loss: a + p + v + n
    out: 
        steps: 205
        piececount: 25
        score_mcts: 0.06
        pacc: 0.93

10  days: 1  
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_playout: 64
    model: a = log_probs - old_log_probs
    loss: a + p + v + n
    out: 
        steps: 242
        piececount: 26
        score_mcts: 0.09
        pacc: 0.94

10.1 days: 3  
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_playout: 64
    model: a = log_probs - old_log_probs
    loss: a + p + v + n
    out: 
        steps: 256
        piececount: 29
        score_mcts: 0.25
        pacc: 0.95

11  days: 1
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_playout: 32
    model: a = log_probs - old_log_probs
    loss: a + v + n
    out: 
        steps: 120
        piececount: 18
        score_mcts: 0.04
        pacc: 0.75

12  days: 1
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_playout: 32
    model: a = log_probs - mcts_log_probs
    loss: a + v + n
    out: 
        steps: 138
        piececount: 20
        score_mcts: 0.04
        pacc: 0.87

13 days: 1
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_playout: 32
    model: a = old_log_probs - log_probs  
    loss: a + v + n + p
    out: 
        steps: 91
        piececount: 9
        score_mcts: 0.04
        pacc: 0.87

14 days: 1
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_playout: 32
    model: a = log_probs - old_log_probs
    loss: a + v + n + p
    out: 
        steps: 151
        piececount: 21
        score_mcts: 0.06
        pacc: 0.8

15  days: 2  
    reward:
        if state.game.terminal: return -2
    v: (q - q_mean)/q_std
    a: ((q_t+1 - q_t) - a_mean)/a_std
    n_playout: 64
    model: a = log_probs - old_log_probs
    loss: a + p + v + n
    out: 
        steps: 
        piececount: 
        score_mcts: 
        pacc:       