# SARSA
# 基于下一个动作的期望值，QL是最优动作

import numpy as np

# Define the Q-table and the learning rate
Q = np.zeros((state_space_size, action_space_size))
alpha = 0.1

# Define the exploration rate and discount factor
epsilon = 0.1
gamma = 0.99

for episode in range(num_episodes):
    current_state = initial_state
    action = epsilon_greedy_policy(epsilon, Q, current_state)
    while not done:
        # Take the action and observe the next state and reward
        next_state, reward, done = take_action(current_state, action)
        # Choose next action using epsilon-greedy policy
        next_action = epsilon_greedy_policy(epsilon, Q, next_state)
        # Update the Q-table using the Bellman equation
        Q[current_state, action] = Q[current_state, action] + alpha * (
                reward + gamma * Q[next_state, next_action] - Q[current_state, action])
        current_state = next_state
        action = next_action