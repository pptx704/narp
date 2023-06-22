import typing
import random
from .algorithms import RandomAlgorithm, QLearning

algo_map = {
    'random': RandomAlgorithm,
    'qlearning': QLearning
}

def get_random_state(client, action: int) -> typing.Tuple[int, float, bool, bool, dict]:
    """
    Returns a random state.
    :param client: The client object
    :type client: narpy.Client
    :param action: The action
    :type action: int
    :return: A random state
    :rtype: typing.Tuple[int, float, bool, bool, dict]
    """
    return random.randrange(action), random.random(), random.choice([True, False]), random.choice([True, False]), {"event": "A random state is sent."}