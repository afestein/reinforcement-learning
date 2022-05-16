import gym
import matplotlib.pyplot as plt
import numpy as np

# Set up environment and parameters
env = gym.make("FrozenLake-v1")
episodes = 1000
successes, total_epochs, total_reward = 0, 0, 0
reward_data = np.array([])

# Repeat the challenge for a specified number of episodes
for _ in range(episodes):
    state = env.reset()
    epoch_count, epoch_reward = 0, 0
    done = False

    # Repeat steps until reaching the destination successfully
    while not done:
        # env.render() # Use to visualise
        action = env.action_space.sample()
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
print(f"Success count: {successes}")
print(f"Fail count: {failures}")
print(f"Success rate: {(successes / episodes) * 100}%")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average rewards per episode: {total_reward / episodes}")

# Plot rewards
episode_data = np.arange(0, episodes, 1)
fig, ax = plt.subplots()
ax.plot(episode_data, reward_data)
ax.set(xlabel="Episode", ylabel="Reward", title="Frozen Lake (random policy)")
ax.grid()
plt.show()

# Plot success/failure rates
fig, ax = plt.subplots()
labels = ["Success", "Failure"]
results = [successes, failures]
ax.bar(labels, results)
ax.set(xlabel="Result", ylabel="Count", title="Frozen Lake (random policy)")
plt.show()
