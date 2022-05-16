# Reinforcement Learning Project

This folder contains various reinforcement learning experiments for
my "Computational Learning" project.

## Contents

The experiments are all based on [OpenAI Gym](https://gym.openai.com).

Three environments were used:

- "Taxi Problem"
- "Cart-Pole"
- "Frozen Lake".

Three algorithm approaches were used:

- Random agent
- Q-Learning
- SARSA

There are three folders that contain the project environments. Inside each
of those is a folder for each algorithm.s

## Environment and Dependencies

Python 3 has been used for development, with [Pipenv](https://pipenv.pypa.io)
as a dependency manager.

Pipenv can be installed using `pip install --user pipenv` or `brew install pipenv`
on MacOS.

To install dependencies locally, use `pipenv install`. This should take care of
all the dependencies for each script.

## Running scripts

To run a specific script it needs to be called with Pipenv so the dependencies
work. For example,

```
pipenv run python3 taxi/sarsa/episodes.py
```

Code output is generally in the form of a text result summary giving the
performance of both the training runs and the trained agent episodes.

Environment visualisations have been disabled by default, but can be re-enabled
by uncommenting the `env.render()` calls where appropriate.
