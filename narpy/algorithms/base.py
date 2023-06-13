class BaseAlgorithm:
    """ 
    Interface for all algorithms. Any algorithm that will be passed to `babopy.Server` must implement this interface.
    """
    def sample(self) -> int:
        """
        Returns an action from the action space.
        :return: The next optimal action from the action space
        :rtype: int
        """
        raise NotImplementedError
    
    def update(self, observation: float, reward: float, termination: bool, truncation: bool, info: dict) -> None:
        """
        Updates the algorithm with the latest observation, reward, termination, truncation and info.
        :param observation: The latest observation
        :type observation: float
        :param reward: The latest reward
        :type reward: float
        :param termination: Whether the episode has terminated
        :type termination: bool
        :param truncation: Whether the episode has been truncated
        :type truncation: bool
        :param info: Additional information
        :type info: dict
        :return: None
        """
        raise NotImplementedError

    def update_action_space(self, action_space: int) -> None:
        """
        Updates the action space.
        :param action_space: The new action space
        :type action_space: int
        :return: None
        """
        self.action_space = action_space
    
    def update_state_space(self, state_space: int) -> None:
        """
        Updates the action space.
        :param action_space: The new action space
        :type action_space: int
        :return: None
        """
        self.state_space = state_space

    def build(self):
        """
        Initiates the algorithm class based on all parameters given previously
        """
        raise NotImplementedError