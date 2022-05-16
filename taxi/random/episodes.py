import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("Taxi-v3").env

total_epochs, total_penalties, total_reward = 0, 0, 0
episodes = 1000
epoch_data = np.array([])
penalty_data = np.array([])
reward_data = np.array([])

# Repeat the challenge for a specified number of episodes
for _ in range(episodes):
    state = env.reset()
    epoch_count, epoch_reward, penalties, reward = 0, 0, 0, 0
    done = False

    # Repeat steps until the passenger is dropped off successfully
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epoch_count += 1
        epoch_reward += reward

    # Print results (use for debugging)
    # print("Timesteps taken: {}".format(epoch_count))
    # print("Penalties incurred: {}".format(penalties))
    # print("Reward earned: {}\n".format(reward))

    # Update the episode tally
    total_epochs += epoch_count
    total_penalties += penalties
    total_reward += epoch_reward

    # Keep track of the episode results for plotting
    epoch_data = np.append(epoch_data, epoch_count)
    penalty_data = np.append(penalty_data, penalties)
    reward_data = np.append(reward_data, epoch_reward)

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
print(f"Average rewards per episode: {total_reward / episodes}")

# Plot the results
episode_data = np.arange(0, episodes, 1)
fig, ax = plt.subplots()
ax.plot(episode_data, reward_data)
ax.set(xlabel="Episode", ylabel="Reward", title="Taxi problem (random policy)")
ax.grid()
plt.show()
