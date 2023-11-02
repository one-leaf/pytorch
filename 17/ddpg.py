# DDPG 
# actor 选择动作
# critic 评估动作

import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

# Define the actor and critic models
actor = Sequential()
actor.add(Dense(32, input_dim=state_space_size, activation='relu'))
actor.add(Dense(32, activation='relu'))
actor.add(Dense(action_space_size, activation='tanh'))
actor.compile(loss='mse', optimizer=Adam(lr=0.001))

critic = Sequential()
critic.add(Dense(32, input_dim=state_space_size, activation='relu'))
critic.add(Dense(32, activation='relu'))
critic.add(Dense(1, activation='linear'))
critic.compile(loss='mse', optimizer=Adam(lr=0.001))

# Define the replay buffer
replay_buffer = []

# Define the exploration noise
exploration_noise = OrnsteinUhlenbeckProcess(size=action_space_size, theta=0.15, mu=0, sigma=0.2)

for episode in range(num_episodes):
    current_state = initial_state
    while not done:
        # Select an action using the actor model and add exploration noise
        action = actor.predict(current_state)[0] + exploration_noise.sample()
        action = np.clip(action, -1, 1)

        # Take the action and observe the next state and reward
        next_state, reward, done = take_action(current_state, action)

        # Add the experience to the replay buffer
        replay_buffer.append((current_state, action, reward, next_state, done))

        # Sample a batch of experiences from the replay buffer
        batch = sample(replay_buffer, batch_size)

        # Update the critic model
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])

        target_q_values = rewards + gamma * critic.predict(next_states)
        critic.train_on_batch(states, target_q_values)

        # Update the actor model
        action_gradients = np.array(critic.get_gradients(states, actions))
        actor.train_on_batch(states, action_gradients)

        current_state = next_state