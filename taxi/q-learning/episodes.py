import gym
import matplotlib.pyplot as plt
import numpy as np
import random
from IPython.display import clear_output

# Set up the environment
env = gym.make("Taxi-v3").env
total_epochs, total_penalties, total_reward = 0, 0, 0
env.reset()

# Set up Q-learning table and hyperparameters
q_table = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Utility arrays for plotting
epoch_data = np.array([])
penalty_data = np.array([])
reward_data = np.array([])

# Train the agent
training_episodes = 100000
for i in range(0, training_episodes):
    state = env.reset()
    epoch_count, episode_reward, penalties, reward = 0, 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epoch_count += 1
        episode_reward += reward

    # Update the episode tally
    total_epochs += epoch_count
    total_penalties += penalties
    total_reward += episode_reward

    # Keep track of the episode results for plotting
    epoch_data = np.append(epoch_data, epoch_count)
    penalty_data = np.append(penalty_data, penalties)
    reward_data = np.append(reward_data, episode_reward)

    if (i + 1) % 1000 == 0:
        clear_output(wait=True)
        print(f"Episode: {i + 1}")

print("Training finished.\n")
print(f"Results after {training_episodes} episodes:")
print(f"Average timesteps per training episode: {total_epochs / training_episodes}")
print(f"Average penalties per training episode: {total_penalties / training_episodes}")
print(f"Average rewards per training episode: {total_reward / training_episodes}\n")

# Run the trained agent against the challenge for specified number of episodes
total_epochs, total_penalties, total_reward = 0, 0, 0
episodes = 1000

for _ in range(episodes):
    state = env.reset()
    epoch_count, episode_reward, penalties, reward = 0, 0, 0, 0
    done = False

    # Repeat steps until the passenger is dropped off successfully
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epoch_count += 1
        episode_reward += reward

        # Print results (use for debugging)
        # print("Timesteps taken: {}".format(epoch_count))
        # print("Penalties incurred: {}".format(penalties))
        # print("Reward earned: {}".format(reward))
        # print("Episode reward: {}\n".format(episode_reward))

    # Update the episode tally
    total_penalties += penalties
    total_epochs += epoch_count
    total_reward += episode_reward

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
print(f"Average rewards per episode: {total_reward / episodes}")

# Plot the training results
episode_data = np.arange(0, training_episodes, 1)
fig, ax = plt.subplots()
ax.plot(episode_data, reward_data)
ax.set(xlabel="Episode", ylabel="Reward", title="Taxi problem (Q-learning)")
ax.grid()
plt.show()
