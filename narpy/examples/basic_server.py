from narpy import make
from pprint import pprint

server = make(seed=42, state_space=True, algorithm='qlearning')

for i in range(10):
    action = server.sample()
    observation, reward, terminated, truncated, info = server.step(action)
    pprint({
        'action': action,
        'observation': observation,
        'reward': reward,
        'terminated': terminated,
        'truncated': truncated,
        'info': info
    })

# stop the server
server.stop()