# narpy
A python library to interact with Arduino agents and abstract Reinforcement Learning procedures.

## Installation
For now, `narpy` does not have a pip package. You can install it by cloning the repository and moving the `narpy` folder to your python site-packages folder or your python project folder.

```bash
git clone git@github.com:zarifikram/narp.git
cd narp
mv narpy /path/to/your/python/site-packages
```

## Usage
The intended use is to have an Arduino agent. However, `narpy` is agnostic of the agent device. It can be used with computer simulations as well.

### Agent
The agent must be able to communicate with `narpy` via TCP/IP protocol. The specific [protocol](#narp-protocol) needs to be followed. `narpy.Client` class is available as an implementation of a python agent.

```python
# Demo client
from narpy import Client
client = Client(5)
client.start()
```

### `narpy` server
Before the agent can start communicate with `narpy`, the server needs to be started. It can be found as `narpy.Server` class. However, a `narpy.make` function is given as a wrapper.

```python
from narpy import make
server = make(seed=42)
for i in range(1000):
    action = server.get_action()
    observation, reward, terminated, truncated, info = server.step(action)
server.stop()
```

### Algorithms
`narpy` provides a set of algorithms that can be used to train the agent. The algorithms can be imported from `narpy.algorithms` module. It is possible to pass a custom algorithm to the `narpy.Server` class but it must follow the `narpy.algorithms.BaseAlgorithm` interface.

### NARP Protocol
TO-DO

## More
Examples can be found in the `examples` folder.

Full documentation can be found [here](https://zarifikram.github.io/Reinforcement-Learning-Based-Obstacle-Detecting-Crawler-Robot/narpy/docs/index.html).
