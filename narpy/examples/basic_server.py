from narpy import make
from pprint import pprint
from time import sleep

server = make(seed=42, state_space=True, algorithm='qlearning')

trajectory = []

for i in range(1000):
    action = server.sample()
    observation, reward, terminated, truncated, info = server.step(action)
    trajectory.append(observation)
    if terminated:
        break
    sleep(0.2)

server.stop()