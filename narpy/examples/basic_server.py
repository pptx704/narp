from narpy import make
from pprint import pprint

server = make(seed=42, state_space=True, algorithm='qlearning')

trajectory = []

for i in range(1000):
    action = server.sample()
    observation, reward, terminated, truncated, info = server.step(action)
    trajectory.append(observation)
    if terminated:
        print(i)
        break
    # pprint({
    #     'action': action,
    #     'observation': observation,
    #     'reward': reward,
    #     'terminated': terminated,
    #     'truncated': truncated,
    #     'info': info
    # })

# stop the server
pprint(trajectory)
server.stop()