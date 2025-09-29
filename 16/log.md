|No     |steps  |piecec |score_m|pacc   |agent  |steps_m|piecec_b   |playout|v           |a                       |model_a             |loss           |cupt   |a_random|
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | -----     | ----- | -----      | -----                  | -----              | -----         | ----- | -----  | 
| xx.1  | 120   | 18    | 0.04  | 0.75  | 2887  | ?     |           | 32    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - old_probs  | a + v - n     | 0.25  | 0.99   |
| xx.1  | 138   | 20    | 0.04  | 0.87  | 3153  | ?     |           | 32    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - mcts_probs | a + v - n     | 0.25  | 0.99   |
| xx.1  | 91    | 9     | 0.04  | 0.87  | 1897  | ?     |           | 32    | q_mean_std | (q_t+1 - q_t)_mean_std | old_probs - probs  | a + p + v - n | 0.25  | 0.99   |
| xx.1  | 151   | 21    | 0.06  | 0.80  | 3633  | ?     |           | 32    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - old_probs  | a + p + v - n | 0.25  | 0.99   |

# 无任何进度
| 0.1  | 96     | 10    | 0.00  | 0.91  | 3808  | 134   | 16    16  | 32    | 0~-1       | (q - v)_mean_std       | probs - old_probs  | a + v - n     | 0.5   | 0.99   |
| 0.2  | 102    | 11    | 0.04  | 0.9   | 5449  | 156   | 18    19  | 64    | 0~-1       | (q - v)_mean_std       | probs - old_probs  | a + v - n     | 0.5   | 0.99   |
| 0.3  | 84     | 8     | 0.1   | 0.92  | 7664  | 171   | 17    20  | 64    | 0~-1       | (q - v)_mean_std       | probs - old_probs  | a + v - n     | 0.5   | 0.99   |

# 训练无进度，并且学到的概率有误
| 1.1   | 183   | 19    | 0.32  | 0.9   | 876   | 228   | -129 -25  | 64    | q_mean_std | (q - v)_mean_std       | mcts_probs - probs | a + p + v - n | 0.5   | 0.99   |
| 1.2   | 217   | 22    | 0.33  | 0.9   | 1684  | 228   | -184 -22  | 64    | q_mean_std | (q - v)_mean_std       | mcts_probs - probs | a + p + v - n | 0.5   | 0.99   |

# 训练恶化，完全不可用，效果更差
| 2.1   | 158   | 21    | 0.08  | 0.9   | 1567  | 160   | -135 -127 | 64    | q_mean_std | (q - v)_mean_std       | probs - mcts_probs | a + p + v - n | 0.5   | 0.99   |
| 3.1   | 147   | 19    | 0.06  | 0.82  | 3234  | 160   | -261 -307 | 32    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - old_probs  | a + p + v - n | 0.25  | 0.99   |

# 0.5 days 失败
| 4.05  | 50    | 10    | 0     | 0.98  | 6851  | 48    | -364 -0   | 32    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - old_probs  | a + p + v - n | 5     | 0.99   |

# cupt 不能过大，否则学习不到任何东西
| 5.02  | 22    | 10    | 0     | 0.94  | 3603  | 26    | -10 -10   | 32    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - old_probs  | a + p + v - n | 5     | 0.9    |
| 5.08  | 110   | 16    | 0.02  | 0.95  | 7287  | 122   | 19   19   | 32    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - old_probs  | a + p + v - n | 1     | 0.99   |

# 效果差
| 6.1   | 50    | 10    | 0.01  | 0.83  | 4172  | 111   | 16   17   | 32    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - old_probs  | a + v - n     | 0.5   | 0.99   |
# 全部动作收敛于down，完全失败
| 9.1   | 7     | 7     | 0     | 0.13  | 2865  | 7     | 7    7    | 32    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - old_probs  | a + v - n     | -     | 0.99   |

# 训练无进度
| 7.1   | 149   | 23    | 0.11  | 0.84  |?      | ?     |           | 64    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - mcts_probs | a + p + v - n | 0.5   | 0.99   |
| 7.2   | 200   | 25    | 0.18  | 0.92  |2983   | ?     | -81 -61   | 64    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - mcts_probs | a + p + v - n | 0.5   | 0.99   |

# 训练有效（快一点）
| 8.1   | 166   | 24    | 0.02  | 0.87  |?      | ?     |           | 64    | q_mean_std | (q_t+1 - q_t)_mean_std | mcts_probs - probs | a + p + v - n | 0.5   | 0.99   |
| 8.3   | 254   | 26    | 0.52  | 0.89  |6240   | 239   | 2124 2149 | 64    | q_mean_std | (q_t+1 - q_t)_mean_std | mcts_probs - probs | a + p + v - n | 0.5   | 0.99   |

# 训练有效
| 10.1  | 242   | 26    | 0.09  | 0.94  |?      | ?     |           | 64    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - old_probs  | a + p + v - n | 0.5   | 0.99   |
| 10.3  | 256   | 29    | 0.25  | 0.95  |4936   | 203   | -254 -56  | 64    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - old_probs  | a + p + v - n | 0.5   | 0.99   |

# 进度缓慢
| 11.1  | 92    | 9     | 0.08  | 0.92  | 2044  | 168   | 18    20  | 64    | 0~-1       | (q_t+1 - q_t)_mean_std | probs - old_probs  | a + v - n     | 0.5   | 0.99   |

# doing
| 12.1  | 92    | 9     | 0.08  | 0.92  | 2044  | 168   | 18    20  | 64    | q_mean_std | (q_t+1 - q_t)_mean_std | probs - old_probs  | a + v - n     | 0.5   | 0.99   |



