# Module for reinforcement learning algorithms
""" 
This module contains different implementations of the reinforcement learning algorithms.

The algorithms available are:
- Random (This is a baseline algorithm that chooses actions randomly)
- Q-Learning (This is a model-free algorithm that uses a Q-Table to choose actions)
"""

from .base import BaseAlgorithm
from .algrandom import RandomAlgorithm
from .qlearning import QLearning