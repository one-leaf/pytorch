# 16/ - 俄罗斯方块 AI（Numba 加速 + PPO + Muon 优化器）

这个目录包含俄罗斯方块游戏的 AI 实现，使用 Numba JIT 加速 MCTS 搜索、PPO 算法和 Muon 正交化优化器，支持 4 动作空间和文件锁状态管理。

## 文件清单

### 核心模块

- **agent.py**: AI 智能体（基础版）
  - 整合 MCTS 和神经网络
  - 动作选择逻辑
  - 自对弈生成训练数据
  - 模型更新接口

- **agent_numba.py**: AI 智能体（Numba 加速版）
  - JIT 编译加速
  - 高性能动作评估
  - 4 动作空间

- **game.py**: 俄罗斯方块游戏环境
  - 棋盘状态管理
  - 方块生成与旋转
  - 碰撞检测
  - 行消除逻辑
  - 游戏结束判断

- **gameenv.py**: 游戏环境封装
  - 标准化接口
  - 状态编码
  - 动作映射
  - 奖励计算

- **mcts.py**: 蒙特卡洛树搜索（基础版）
  - 选择：UCB1 公式
  - 扩展：神经网络预测先验概率
  - 模拟：快速 rollout 或使用网络
  - 回溯：更新节点统计信息

- **mcts_single.py**: 单线程 MCTS
  - 简化版 MCTS 实现
  - 用于调试和测试

- **mcts_single_numba.py**: 单线程 MCTS（Numba 加速）
  - JIT 编译加速
  - 高性能搜索

- **model.py**: 神经网络模型
  - 特征提取层
  - 策略头：输出动作概率分布
  - 价值头：输出局面评估

- **muon.py**: Muon 优化器
  - 正交化优化器
  - 矩阵归一化
  - 替代 Adam/SGD

- **selfplay.py**: 自对弈引擎（版本 1）
  - MCTS 指导的自我对弈
  - 生成 (state, policy, value) 三元组
  - 数据保存到文件

- **selfplay2.py**: 自对弈引擎（版本 2）
  - 改进的自对弈逻辑
  - 不同的探索策略

- **selfplay3.py**: 自对弈引擎（版本 3）
  - 进一步优化
  - PPO 算法集成

- **status.py**: 状态管理
  - 文件锁状态管理
  - 进程间通信
  - 训练状态同步

- **train.py**: 训练循环
  - 加载自对弈数据
  - 策略损失 + 价值损失
  - 数据增强
  - 模型检查点保存
  - PPO 更新

- **test.py**: 模型评估
  - 与旧版本对战
  - 胜率统计
  - 性能指标记录

- **vit.py**: Vision Transformer 实现
  - Patch Embedding
  - Multi-Head Self-Attention
  - Transformer Encoder
  - Position Embedding

- **removedata.py**: 数据清理
  - 清理旧训练数据
  - 释放磁盘空间

### 数据目录

- **data/**: 自对弈生成的训练数据
- **model/**: 训练好的模型检查点
