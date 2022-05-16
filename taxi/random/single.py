import gym
import matplotlib.pyplot as plt
import numpy as np

# Set up environment
env = gym.make("Taxi-v3").env
env.reset()
env.render()

epoch_count, penalties, reward = 0, 0, 0
rewards = np.array([])
cumulative_rewards = np.array([])
done = False

# Repeat steps until the passenger is dropped off successfully
while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    epoch_count += 1
    rewards = np.append(rewards, reward)
    cumulative_reward = np.sum(rewards)
    cumulative_rewards = np.append(cumulative_rewards, cumulative_reward)

print("Timesteps taken: {}".format(epoch_count))
print("Penalties incurred: {}".format(penalties))
print("Reward earned: {}".format(cumulative_reward))

# Plot the results
epochs = np.arange(0, epoch_count, 1)
fig, ax = plt.subplots()
ax.plot(epochs, cumulative_rewards)
ax.set(xlabel="Epoch", ylabel="Reward", title="Taxi problem (random policy)")
ax.grid()
plt.show()
