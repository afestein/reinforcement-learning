# Frozen Lake reinforcement learning problem with Q-learning
# Code adapted from https://github.com/simoninithomas/Deep_reinforcement_learning_Course/
import gym
import matplotlib.pyplot as plt
import numpy as np
import random

# Set up environment and parameters
env = gym.make("FrozenLake-v1")
successes, total_epochs, total_reward = 0, 0, 0
reward_data = np.array([])

# Set up Q-learning table and hyperparameters
q_table = np.zeros([env.observation_space.n, env.action_space.n])
learning_rate = 0.8
discount = 0.95
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

# Train the agent
training_episodes = 100000
for episode in range(training_episodes):
    state = env.reset()
    epoch_count, epoch_reward = 0, 0
    done = False

    # Repeat steps until reaching the destination successfully
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Make a random move
        else:
            action = np.argmax(q_table[state])  # Use trained Q-table values

        new_state, reward, done, info = env.step(action)

        # Update the Q-table
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount * np.max(q_table[new_state, :]) - q_table[state, action]
        )

        state = new_state

        if reward == 1:
            successes += 1

        epoch_count += 1
        epoch_reward += reward

    # Update and decay the epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    # Update the episode tally
    total_epochs += epoch_count
    total_reward += epoch_reward

    # Keep track of the episode results for plotting
    reward_data = np.append(reward_data, epoch_reward)

    # Print results (use for debugging)
    # print("Timesteps taken: {}".format(epoch_count))
    # print("Reward earned: {}\n".format(reward))

failures = training_episodes - successes
print("Training results:")
print(f"Success count: {successes}")
print(f"Fail count: {failures}")
print(f"Success rate: {(successes / training_episodes) * 100}%")
print(f"Average timesteps per episode: {total_epochs / training_episodes}")
print(f"Average rewards per episode: {total_reward / training_episodes}")

# # Plot training results
# episode_data = np.arange(0, training_episodes, 1)
# fig, ax = plt.subplots()
# ax.plot(episode_data, reward_data)
# ax.set(xlabel="Episode", ylabel="Reward", title="Frozen Lake (Q-Learning)")
# ax.grid()
# plt.show()

# # Plot training success/failure rates
# fig, ax = plt.subplots()
# labels = ["Success", "Failure"]
# results = [successes, failures]
# ax.bar(labels, results)
# ax.set(xlabel="Result", ylabel="Count", title="Frozen Lake (Q-Learning)")
# plt.show()

# Run the trained agent against the challenge for specified number of episodes
env.reset()
episodes = 1000
successes, total_epochs, total_reward = 0, 0, 0
reward_data = np.array([])

for episode in range(episodes):
    state = env.reset()
    epoch_count, epoch_reward = 0, 0
    done = False

    # Repeat steps until reaching the destination successfully
    while not done:
        # env.render() # Use to visualise

        # Use trained Q-table values
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == 1:
            successes += 1

        epoch_count += 1
        epoch_reward += reward

    # Update the episode tally
    total_epochs += epoch_count
    total_reward += epoch_reward

    # Keep track of the episode results for plotting
    reward_data = np.append(reward_data, epoch_reward)

    # Print results (use for debugging)
    # print("Timesteps taken: {}".format(epoch_count))
    # print("Reward earned: {}\n".format(reward))

failures = episodes - successes
print("\nAgent results:")
print(f"Success count: {successes}")
print(f"Fail count: {failures}")
print(f"Success rate: {(successes / episodes) * 100}%")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average rewards per episode: {total_reward / episodes}")

# Plot results
episode_data = np.arange(0, episodes, 1)
fig, ax = plt.subplots()
ax.plot(episode_data, reward_data)
ax.set(xlabel="Episode", ylabel="Reward", title="Frozen Lake (Q-Learning)")
ax.grid()
plt.show()

# Plot success/failure rates
fig, ax = plt.subplots()
labels = ["Success", "Failure"]
results = [successes, failures]
ax.bar(labels, results)
ax.set(xlabel="Result", ylabel="Count", title="Frozen Lake (Q-Learning)")
plt.show()
