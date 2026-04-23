# 17/ - 强化学习基础教程（Hands-on-RL）

这个目录是《动手学强化学习》（Hands-on-RL）项目的代码示例，涵盖了多臂老虎机、马尔可夫决策过程、动态规划等强化学习基础概念和算法实现。

## 文件清单

### 核心模块

- **2.py**: 多臂老虎机问题 - 伯努利多臂老虎机及探索策略
  - `BernoulliBandit` 类：伯努利多臂老虎机
    - `probs`: 随机生成 K 个 0~1 的获奖概率
    - `best_idx`: 获奖概率最大的拉杆索引
    - `best_prob`: 最大获奖概率
    - `step(k)`: 拉动 k 号拉杆，返回 1（获奖）或 0（未获奖）
  - `Solver` 类：多臂老虎机算法基本框架
    - `counts`: 每根拉杆的尝试次数
    - `regret`: 累积懊悔（选择非最优拉杆的次数）
    - `actions`: 记录每一步的动作
    - `regrets`: 记录每一步的累积懊悔
    - `update_regret(k)`: 计算累积懊悔（`1 if best_idx != k else 0`）
    - `run(num_steps)`: 运行指定次数
  - `EpsilonGreedy` 类：ε-贪婪算法
    - `epsilon`: 探索概率（默认 0.01）
    - `estimates`: 每根拉杆的期望奖励估值（初始化为 1.0）
    - `run_one_step()`: ε 概率随机选择，否则选择估值最大的拉杆
    - 更新公式：`estimates[k] += 1/(counts[k]+1) * (r - estimates[k])`
  - `plot_results()`: 绘制累积懊悔随时间变化的图像
  - 实验设置：K=10 臂老虎机，运行 5000 步，测试不同 ε 值（1e-4, 0.01, 0.1, 0.25, 0.5）
  - 参考：https://hrl.boyuai.com/chapter/1/多臂老虎机

- **3.py**: 马尔可夫决策过程 - MRP 和 MDP 的状态价值计算
  - **MRP（马尔可夫奖励过程）**：
    - `P`: 6×6 状态转移概率矩阵
    - `rewards`: [-1, -2, -2, 10, 1, 0]
    - `gamma`: 折扣因子 0.5
    - `compute_return()`: 计算从起始状态到终止状态的回报
      - 公式：`G = γ * G + rewards[chain[i] - 1]`
      - 示例：`chain = [1, 2, 3, 6]`，回报 = -2.5
    - `compute()`: 贝尔曼方程矩阵形式求解状态价值
      - 公式：`V = (I - γP)^(-1) * R`
      - 结果：6 个状态的价值分别为 [-2.02, -2.21, 1.16, 10.54, 3.59, 0]
  - **MDP（马尔可夫决策过程）**：
    - `S`: 状态集合 ["s1", "s2", "s3", "s4", "s5"]
    - `A`: 动作集合 ["保持 s1", "前往 s1", ..., "概率前往"]
    - `P`: 状态转移函数字典（如 `"s1-保持 s1-s1": 1.0`）
    - `R`: 奖励函数字典（如 `"s1-保持 s1": -1`）
    - `Pi_1`: 随机策略（所有动作概率 0.5）
    - `Pi_2`: 偏好策略（不同动作有不同概率）
  - 参考：https://hrl.boyuai.com/chapter/1/马尔可夫决策过程

- **4.py**: 动态规划算法 - 策略迭代和值迭代
  - `CliffWalkingEnv` 类：悬崖漫步环境
    - `ncol=12`, `nrow=4`: 4×12 网格世界
    - `P[state][action]`: 转移矩阵 `[(p, next_state, reward, done)]`
    - 4 种动作：上、下、左、右
    - 悬崖奖励：-100（掉下悬崖）
    - 每步奖励：-1
    - 目标：从左上角走到右下角
  - `PolicyIteration` 类：策略迭代算法
    - `v`: 状态价值初始化（全 0）
    - `pi`: 策略初始化（均匀随机 0.25）
    - `theta`: 策略评估收敛阈值
    - `policy_evaluation()`: 策略评估
      - 计算所有 Q(s,a) 价值
      - 更新状态价值：`new_v[s] = sum(pi[s][a] * qsa)`
      - 迭代直到 `max_diff < theta`
    - `policy_improvement()`: 策略提升
      - 选择 Q 值最大的动作
      - 多个最优动作均分概率
    - `policy_iteration()`: 交替执行评估和提升，直到策略收敛
  - `print_agent()`: 打印智能体的状态价值和策略
  - 参考：https://hrl.boyuai.com/chapter/1/动态规划算法

- **rl_utils.py**: 强化学习工具库 - 通用训练函数和辅助工具
  - `ReplayBuffer` 类：经验回放缓冲区
    - `capacity`: 最大容量（使用 `collections.deque(maxlen=capacity)`）
    - `add(state, action, reward, next_state, done)`: 添加转移数据
    - `sample(batch_size)`: 随机采样一批数据
    - `size()`: 返回当前数据量
  - `moving_average(a, window_size)`: 计算滑动平均（平滑曲线）
    - 使用 `np.cumsum()` 高效计算
    - 处理开始、中间、结束部分
  - `train_on_policy_agent(env, agent, num_episodes)`: 在线策略训练
    - 每轮收集完整轨迹后更新（如 REINFORCE、A2C）
    - `transition_dict`: 存储 states, actions, next_states, rewards, dones
    - `agent.update(transition_dict)`: 使用完整轨迹更新
    - 使用 `tqdm` 显示训练进度
  - `train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)`: 离线策略训练
    - 每步存入缓冲区，采样小批量更新（如 DQN、DDPG）
    - `replay_buffer.add()`: 添加转移数据
    - `replay_buffer.sample()`: 采样训练
  - 依赖：`tqdm`, `numpy`, `torch`, `collections`, `random`

## 来源

https://github.com/boyu-ai/Hands-on-RL/