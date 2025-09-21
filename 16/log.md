|No     |steps  |piecec |score_m|pacc   |agent  |steps_m|win_lost   |playout|v                   |a                               |model_a             |loss           |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | -----     | ----- | -----              | -----                          | -----              | -----         |
| xx.1  | 120   | 18    | 0.04  | 0.75  |2887   | ?     |           | 32    | (q - q_mean)/q_std | ((q_t+1 - q_t) - a_mean)/a_std | probs - old_probs  | a + v - n     |
| xx.1  | 138   | 20    | 0.04  | 0.87  |3153   | ?     |           | 32    | (q - q_mean)/q_std | ((q_t+1 - q_t) - a_mean)/a_std | probs - mcts_probs | a + v - n     |
| xx.1  | 91    | 9     | 0.04  | 0.87  |1897   | ?     |           | 32    | (q - q_mean)/q_std | ((q_t+1 - q_t) - a_mean)/a_std | old_probs - probs  | a + p + v - n |
| xx.1  | 151   | 21    | 0.06  | 0.80  |3633   | ?     |           | 32    | (q - q_mean)/q_std | ((q_t+1 - q_t) - a_mean)/a_std | probs - old_probs  | a + p + v - n |

# 训练无进度，并且学到的概率有误
| 1.1   | 183   | 19    | 0.32  | 0.9   | 876   | 228   | -129 -25  | 64    | (q - q_mean)/q_std | ((q - v) - a_mean)/a_std       | mcts_probs - probs | a + p + v - n |
| 1.2   | 217   | 22    | 0.33  | 0.9   | 1684  | 228   | -184 -22  | 64    | (q - q_mean)/q_std | ((q - v) - a_mean)/a_std       | mcts_probs - probs | a + p + v - n |

# 训练恶化，完全不可用，效果更差
| 2.1   | 158   | 21    | 0.08  | 0.9   | 1567  | 160   | -135 -127 | 64    | (q - q_mean)/q_std | ((q - v) - a_mean)/a_std       | probs - mcts_probs | a + p + v - n |

# doing
| 3.1   | 242   | 26    | 0.09  | 0.94  |?      | ?     |           | 32    | (q - q_mean)/q_std | ((q_t+1 - q_t) - a_mean)/a_std | probs - old_probs  | a + p + v - n |


# 训练无进度
| 5.1   | 149   | 23    | 0.11  | 0.84  |?      | ?     |           | 64    | (q - q_mean)/q_std | ((q_t+1 - q_t) - a_mean)/a_std | probs - mcts_probs | a + p + v - n |
| 5.2   | 200   | 25    | 0.18  | 0.92  |2983   | ?     | -81 -61   | 64    | (q - q_mean)/q_std | ((q_t+1 - q_t) - a_mean)/a_std | probs - mcts_probs | a + p + v - n |

# 训练有效（快一点）
| 8.1   | 166   | 24    | 0.02  | 0.87  |?      | ?     |           | 64    | (q - q_mean)/q_std | ((q_t+1 - q_t) - a_mean)/a_std | mcts_probs - probs | a + p + v - n |
| 8.3   | 254   | 26    | 0.52  | 0.89  |6240   | 239   | 2124 2149 | 64    | (q - q_mean)/q_std | ((q_t+1 - q_t) - a_mean)/a_std | mcts_probs - probs | a + p + v - n |

# 训练有效
| 10.1  | 242   | 26    | 0.09  | 0.94  |?      | ?     |           | 64    | (q - q_mean)/q_std | ((q_t+1 - q_t) - a_mean)/a_std | probs - old_probs  | a + p + v - n |
| 10.3  | 256   | 29    | 0.25  | 0.95  |4936   | 203   | -254 -56  | 64    | (q - q_mean)/q_std | ((q_t+1 - q_t) - a_mean)/a_std | probs - old_probs  | a + p + v - n |





