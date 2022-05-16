# CartPole problem Q-learning solution
# Code adapted from https://github.com/JackFurby/CartPole-v0
# and https://medium.com/swlh/using-q-learning-for-openais-cartpole-v1-4a216ef237df

import gym
import matplotlib.pyplot as plt
import numpy as np

# Set up environment and parameters
env = gym.make("CartPole-v1")
env._max_episode_steps = 200
reward_data = np.array([])
training_episodes = 10000
learning_rate = 0.1
discount = 0.95
total_reward = 0

# Set up bins and Q-table
numBins = 20
obsSpaceSize = len(env.observation_space.high)
# Bin ranges are taken from the observation space sizes available at
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
bins = [
    np.linspace(-4.8, 4.8, numBins),
    np.linspace(-4, 4, numBins),
    np.linspace(-0.418, 0.418, numBins),
    np.linspace(-4, 4, numBins),
]
q_table = np.random.uniform(
    low=-2, high=0, size=([numBins] * obsSpaceSize + [env.action_space.n])
)

# Exploration settings
epsilon = 1  # Variable to decide how often to pick a random action (will be decayed)
start_epsilon_decaying = 1
end_epsilon_decaying = training_episodes
epsilon_decay_value = epsilon / (training_episodes - start_epsilon_decaying)

# Assign a state to a discrete state index in the q_table
def get_discrete_state(state):
    stateIndex = []
    for i in range(obsSpaceSize):
        stateIndex.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(stateIndex)


for episode in range(training_episodes):
    discrete_state = get_discrete_state(env.reset())
    done = False
    episode_reward = 0
    epoch_count = 0

    while not done:
        epoch_count += 1
        # Use previously explored Q-table value
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        # Pick a random action
        else:
            action = np.random.randint(0, env.action_space.n)

        # Perform the action
        observation, reward, done, info = env.step(action)
        episode_reward += reward

        # Calculate Q values and update the Q-table
        new_discrete_state = get_discrete_state(observation)
        current_q = q_table[discrete_state + (action,)]

        # Estimate optimal future value
        max_future_q = np.max(q_table[new_discrete_state])

        # Penalise training agent if it failed to balance the pole before maximum epochs
        if done and epoch_count < 200:
            reward = -375

        new_q = (1 - learning_rate) * current_q + learning_rate * (
            reward + discount * max_future_q
        )
        q_table[discrete_state + (action,)] = new_q
        discrete_state = new_discrete_state

    # print("Episode reward: {}\n".format(episode_reward))

    # Update and decay the epsilon
    if end_epsilon_decaying >= episode >= start_epsilon_decaying:
        epsilon -= epsilon_decay_value

    # Keep track of the episode results
    reward_data = np.append(reward_data, episode_reward)
    total_reward += episode_reward

print(f"Average training rewards per episode: {total_reward / training_episodes}")

# Run the trained agent against the challenge for specified number of episodes
env.reset()
episodes = 1000
total_reward = 0
for _ in range(episodes):
    discrete_state = get_discrete_state(env.reset())
    done = False
    episode_reward = 0

    while not done:
        # env.render() # Use for visualisation when required
        # Use trained Q-table values
        action = np.argmax(q_table[discrete_state])

        # Perform the action
        observation, reward, done, info = env.step(action)
        episode_reward += reward

        # Update the discrete state
        new_discrete_state = get_discrete_state(observation)
        discrete_state = new_discrete_state

    # print("Episode reward: {}\n".format(episode_reward))

    # Keep track of the episode results
    total_reward += episode_reward

print(f"Average trained agent rewards per episode: {total_reward / episodes}")

# Plot the training results
episode_data = np.arange(0, training_episodes, 1)
fig, ax = plt.subplots()
ax.plot(episode_data, reward_data)
ax.set(xlabel="Episode", ylabel="Reward", title="Cart-Pole (Q-Learning)")
ax.grid()
plt.show()
