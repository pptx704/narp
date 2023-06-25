import random
import typing

from .base import BaseAlgorithm

class RandomAlgorithm(BaseAlgorithm):
    """
    This algorithm chooses an action from the action space randomly.
    """
    def __init__(self, action_space: int = -1, state_space=None, **kwargs):
        """
        Initializes the algorithm.
        :param action_space: The action space
        :type action_space: int
        :param state_space: The state space. Not used in this algorithm
        :type state_space: None
        """
        super().__init__(**kwargs)
        self.action_space = action_space
        self.state_space = state_space
        
    def sample(self) -> int:
        """
        Returns an action from the action space.
        :return: A random action from the action space
        :rtype: int
        """
        return random.randrange(start=0, stop=self.action_space)

    def update(self, *args, **kwargs) -> None:
        """ 
        For interface compatibility. Does nothing since it only returns a random action.
        """
        super().update(*args, **kwargs)
        return

    def build(self) -> None:
        """ 
        For interface compatibility. Does nothing since it only returns a random action.
        """
        pass