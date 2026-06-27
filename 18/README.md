# 18/ - 俄罗斯方块 AI（GRPO）

这个目录包含俄罗斯方块游戏的 AI 实现，使用 GRPO（Group Relative Policy Optimization）算法进行训练。

## 文件清单

### 核心模块

- **agent.py**: AI 智能体
  - 游戏状态管理（棋盘、方块、分数）
  - 动作选择逻辑（5 动作空间）
  - 状态编码输出 [4, 20, 10]
  - save_state/restore_from_state 用于 GRPO fork rollout

- **model.py**: 策略价值网络
  - 基于 Decoder-only Transformer 架构（`transformer.py`）
  - 策略头：输出动作 log 概率 [B, 5]
  - 无价值头（纯 GRPO，value 返回 dummy zeros）
  - train_step_grpo: GRPO 训练方法

- **selfplay.py**: GRPO 数据采集引擎
  - 用当前策略运行 G 局游戏
  - 记录轨迹 (state, action, ref_prob)
  - 按组标准化计算优势

- **train.py**: GRPO 训练循环
  - 加载自对弈数据
  - PPO clip 策略损失 + KL 惩罚 + 熵正则
  - 模型检查点保存

- **augment.py**: 数据增强工具
  - 左右翻转（swap LEFT/RIGHT 动作和概率）
  - selfplay 和 train 共享

- **vit.py**: Vision Transformer 实现
  - Patch Embedding
  - Multi-Head Self-Attention
  - Transformer Encoder
  - 双 token（action + value）

- **status.py**: 状态管理
  - 文件锁状态管理
  - 训练状态同步

### 数据目录

- **data/**: 自对弈生成的训练数据
- **model/**: 训练好的模型检查点
