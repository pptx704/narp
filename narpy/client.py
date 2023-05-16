import socket
import typing
import json 
from .utils import get_random_state

class Client:
    """
    Client implementation of a python client that following the babopy protocol.
    """
    def __init__(self, action_space: int, host: str = 'localhost', port: int = 7234, func: typing.Callable = None):
        """
        Initializes the client.
        Server must be running before the client can connect.
        :param action_space: The action space to use. 
        :type action_space: int
        :param host: The host to connect to. Defaults to localhost.
        :type host: str
        :param port: The port to connect to. Defaults to 7234.
        :type port: int
        :param func: The function to use to generate a random state. Defaults to babopy.utils.get_random_state. Function signature must be func(action: int) -> typing.Tuple[int, float, bool, bool, dict]
        :type func: typing.Callable
        """
        self.host = host
        self.port = port
        self.action_space = action_space
        if func is not None:
            self.func = func
        else:
            self.func = get_random_state

    def send(self, data) -> None:
        """
        This is a helper function. Takes a string as an argument and encodes it to bytes before sending to server.
        :param data: The data to send
        :type data: str
        :return: None
        """
        if not isinstance(data, str):
            raise TypeError('data must be a string')
        self.socket.send(data.encode())

    def receive(self, size: int = 1024) -> str:
        """
        This is a helper function. Receives data from server and decodes it to string before returning.
        """
        return self.socket.recv(size).decode()

    def make_data(self, data_dict: dict) -> str:
        """
        This is a helper function to convert a dictionary to a NARP compliant string.
        """
        string = ""
        string.add(f"OBS {data_dict['observation']}\n")
        string.add(f"REW {data_dict['reward']}\n")
        string.add(f"TER {data_dict['terminated']}\n")
        string.add(f"TRN {data_dict['truncated']}\n")
        for key, value in data_dict['info'].items():
            string.add(f"INF {key} {value}\n")
        string += "END"
        return string


    def start(self) -> None:
        """
        Starts the client. This is a blocking call. This method will not return until the server disconnects or sends the 'END' command.
        For using it as a part of a larger program, it is recommended to run it in a separate thread.
        :return: None
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        try:
            while True:
                data = self.receive()
                if data == 'END':
                    self.close()
                    break
                elif data == 'ASP':
                    self.send(str(self.action_space))
                elif data.startswith('ACT'):
                    action = int(data.split(' ')[1])
                    state = self.func(action)
                    self.send(self.make_data({
                        'observation': state[0],
                        'reward': state[1],
                        'terminated': state[2],
                        'truncated': state[3],
                        'info': state[4]
                    }))
                else:
                    raise ValueError(f'Invalid command: {data}')
        except KeyboardInterrupt:
            # Keep in mind that receiving is blocking. 
            # So if the user presses Ctrl+C, the client will not close until it receives a command from the server.
            self.close()
        except Exception as e:
            self.close()
            raise e

    def close(self):
        """
        Closes the connection to the server.
        :return: None
        """
        self.socket.close()