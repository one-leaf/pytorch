# 12/ - 俄罗斯方块 AI（Vision Transformer 版）

这个目录包含俄罗斯方块游戏的 AI 实现，使用 Vision Transformer（ViT）架构进行策略和价值预测，支持双游戏并行自对弈和动态 c_puct 调整。

## 文件清单

### 核心模块

- **agent.py**: AI 智能体
  - 整合 MCTS 和 ViT 网络
  - 动作选择逻辑
  - 自对弈生成训练数据
  - 模型更新接口

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

- **mcts.py**: 蒙特卡洛树搜索
  - 选择：UCB1 公式
  - 扩展：神经网络预测先验概率
  - 模拟：快速 rollout 或使用网络
  - 回溯：更新节点统计信息
  - `c_puct`: 动态探索常数调整
  - `n_playout`: 模拟次数
  - 帧状态追踪

- **model.py**: 神经网络模型
  - 特征提取层
  - 策略头：输出动作概率分布
  - 价值头：输出局面评估

- **selfplay.py**: 自对弈引擎（版本 1）
  - 双游戏并行自对弈
  - 生成 (state, policy, value) 三元组
  - 数据保存到文件
  - 温度参数控制探索
  - 动态 c_puct 调整

- **selfplay2.py**: 自对弈引擎（版本 2）
  - 改进的自对弈逻辑
  - 不同的探索策略

- **train.py**: 训练循环
  - 加载自对弈数据
  - 策略损失 + 价值损失
  - 数据增强
  - 模型检查点保存

- **test.py**: 模型评估
  - 与旧版本对战
  - 胜率统计
  - 性能指标记录

- **vit.py**: Vision Transformer 实现
  - Patch Embedding
  - Multi-Head Self-Attention
  - Transformer Encoder
  - Position Embedding
  - ViT-Ti/S/B 多尺寸配置

### 数据目录

- **data/**: 自对弈生成的训练数据
- **model/**: 训练好的模型检查点
