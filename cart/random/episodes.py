import gym
import matplotlib.pyplot as plt
import numpy as np

# Set up environment and parameters
env = gym.make("CartPole-v1")
reward_data = np.array([])
episodes = 1000
total_reward = 0

# Run multiple episodes and track average results
for _ in range(episodes):
    env.reset()
    episode_reward = 0
    done = False

    # Keep balancing the pole as long as possible
    while not done:
        # env.render()
        action = env.action_space.sample()  # Random action
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        # print("Episode reward: {}\n".format(episode_reward))

    # Keep track of the episode results for plotting
    reward_data = np.append(reward_data, episode_reward)
    total_reward += episode_reward

print(f"Average rewards per episode: {total_reward / episodes}")

# Plot the results
episode_data = np.arange(0, episodes, 1)
fig, ax = plt.subplots()
ax.plot(episode_data, reward_data)
ax.set(xlabel="Episode", ylabel="Reward", title="Cart-Pole (random)")
ax.grid()
plt.show()
