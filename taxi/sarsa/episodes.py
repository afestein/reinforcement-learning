# Code modified from https://github.com/MrShininnnnn/SARSA-Frozen-Lake/

import gym
import numpy as np
import matplotlib.pyplot as plt

# Utility function for 'Epsilon greedy' exploration strategy
def epsilon_greedy(q_table, epsilon, state):
    # selects a random action with probability epsilon
    if np.random.random() <= epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(q_table[state])


# Set up environment and parameters
env = gym.make("Taxi-v3").env
total_epochs, total_penalties, total_reward = 0, 0, 0
reward_data = np.array([])
n_states, n_actions = (env.observation_space.n, env.action_space.n)

# Set up Q-learning table and hyperparameters
q_table = np.zeros((n_states, n_actions))
alpha = 0.1
gamma = 0.9
epsilon = 0.5

# Train the agent
training_episodes = 100000
for episode in range(training_episodes):
    state = env.reset()
    action = epsilon_greedy(q_table, epsilon, state)
    done = False
    epoch_count, episode_reward, penalties, reward = 0, 0, 0, 0

    # Repeat steps until reaching the destination successfully
    while not done:
        new_state, reward, done, info = env.step(action)
        new_action = epsilon_greedy(q_table, epsilon, new_state)

        # Update the Q-table
        q_table[state, action] += alpha * (
            reward + (gamma * q_table[new_state, new_action]) - q_table[state, action]
        )

        if reward == -10:
            penalties += 1

        epoch_count += 1
        episode_reward += reward

        state, action = new_state, new_action

    # Update the episode tally
    total_epochs += epoch_count
    total_penalties += penalties
    total_reward += episode_reward

    # Keep track of the episode results for plotting
    reward_data = np.append(reward_data, episode_reward)

print("Training finished.\n")
print(f"Results after {training_episodes} episodes:")
print(f"Average timesteps per training episode: {total_epochs / training_episodes}")
print(f"Average penalties per training episode: {total_penalties / training_episodes}")
print(f"Average rewards per training episode: {total_reward / training_episodes}\n")

# Run the trained agent against the challenge for specified number of episodes
env.reset()
episodes = 1000
total_epochs, total_penalties, total_reward = 0, 0, 0

for episode in range(episodes):
    state = env.reset()
    epoch_count, episode_reward, penalties, reward = 0, 0, 0, 0
    done = False
    # Fetch an action from the Q-table
    action = np.argmax(q_table[state])

    # Repeat steps until reaching the destination successfully
    while not done:
        # env.render() # Use to visualise
        # take actions according the state and trained Q-table
        new_state, reward, done, info = env.step(action)
        new_action = np.argmax(q_table[new_state])

        if reward == -10:
            penalties += 1

        epoch_count += 1
        episode_reward += reward

        # Exit the script if it takes too long
        if epoch_count >= 2500:
            done = True

        state, action = new_state, new_action

    # Update the episode tally
    total_penalties += penalties
    total_epochs += epoch_count
    total_reward += episode_reward

    # Print results (use for debugging)
    # print("Timesteps taken: {}".format(epoch_count))
    # print("Reward earned: {}\n".format(reward))

env.close()

print(f"Trained agent results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
print(f"Average rewards per episode: {total_reward / episodes}")

# Plot the training results
episode_data = np.arange(0, training_episodes, 1)
fig, ax = plt.subplots()
ax.plot(episode_data, reward_data)
ax.set(xlabel="Episode", ylabel="Reward", title="Taxi problem (SARSA)")
ax.grid()
plt.show()
