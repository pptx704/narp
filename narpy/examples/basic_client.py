import narpy
from pprint import pprint

def get_and_print_random_client(action):
    state = narpy.utils.get_random_state(action)
    pprint({
        'action': action,
        'state': state
    }) 
    return state

client = narpy.Client(action_space=5, func=get_and_print_random_client)
client.start()