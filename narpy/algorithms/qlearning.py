import random
import typing
import numpy as np

from .base import BaseAlgorithm

class QLearning(BaseAlgorithm):
    """
    This algorithm chooses an action based on the Q-Table.
    """
    def __init__(
            self, 
            action_space: int = -1, 
            state_space=None, 
            learning_rate=0.1, 
            discount_factor=0.9, 
            min_epsilon=0.05,
            max_epsilon=1.0,
            epsilon_decay=0.01
        ):
        """
        Initializes the algorithm.
        :param action_space: The action space
        :type action_space: int
        :param state_space: The state space. Not used in this algorithm
        :type state_space: None
        """
        self.action_space = action_space
        self.state_space = state_space
        self.current_state = 0
        self.current_reward = None
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.current_action = None

    def getBestActionInAState(self, state):
        """
        Returns the best action in a state using the Q-Table.
        :param state: The state
        :type state: int
        :return: The best action in a state
        :rtype: int
        """
        return np.argmax(self.qtable[state, :])
    
    def epsilonGreedyPolicy(self, state, epsilon) -> int:
        """
        Returns an action based on the Epsilon Greedy Policy.
        Generates a random number between 0 and 1. If the number is less than epsilon, it returns a random action.
        Otherwise, it returns the best action in the state.

        :param state: The state
        :type state: int
        :param epsilon: The epsilon value
        :type epsilon: float
        :return: An action based on Epsilon Greedy Policy
        :rtype: int
        """
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            return self.getBestActionInAState(state)
        
    def update_action_space(self, action_space: int) -> None:
        """
        Updates the action space.
        :param action_space: The new action space
        :type action_space: int
        :return: None
        """
        self.action_space = action_space
    
    def get_epsilon(self) -> float:
        """
        Returns the epsilon value.
        :return: The epsilon value
        :rtype: float
        """
        return max(self.max_epsilon - self.epsilon_decay * self.current_state, self.min_epsilon)

    def sample(self) -> int:
        """
        Returns an action from the action space.
        :return: An action based on Epsilon Greedy Policy from the action space
        :rtype: int
        """
        self.current_action = self.epsilonGreedyPolicy(self.current_state, self.get_epsilon())
        return self.current_action

    def update(self, observation: float, reward: float, termination: bool, truncation: bool, info: dict) -> None:
        """ 
        Updates the qtable based on the reward and the observation.
        :param observation: The observation
        :type observation: float
        :param reward: The reward
        :type reward: float
        :param termination: Whether the episode has terminated
        :type termination: bool
        :param truncation: Whether the episode has been truncated
        :type truncation: bool
        :param info: Additional information
        :type info: dict
        :return: None
        """
        td = reward + self.discount_factor * np.max(self.qtable[observation, :]) - self.qtable[self.current_state, self.current_action]
        self.qtable[self.current_state, self.current_action] += self.learning_rate * td
        self.current_reward = reward
        self.current_state = observation

    def build(self):
        """ 
        For interface compatibility. Does nothing since it only returns a random action.
        """
        try:
            self.qtable = np.zeros((self.state_space, self.action_space))
        except TypeError:
            raise TypeError("Q-Learning requires a state space that is an integer. Try \"make(*args, state_space=True)\" to get the state space.")