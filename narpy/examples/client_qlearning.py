from narpy import client

def generate_data(client, action):
    # task
    # Reaching from 0, 0 of a 2D plane to 4, 4
    # Action space: 5
    # 0: up
    # 1: down
    # 2: left
    # 3: right
    # 4: stay
    # State space: 25 - We need to go from 0 to 24
    print(action)
    print(client.data)
    position = client.data['position']
    distance = client.data['distance']
    # convert position to x, y
    point = (position % 5, position // 5)
    if action == 0:
        point = (point[0], max(0, point[1] - 1))
    elif action == 1:
        point = (point[0], min(4, point[1] + 1))
    elif action == 2:
        point = (max(0, point[0] - 1), point[1])
    elif action == 3:
        point = (min(4, point[0] + 1), point[1])
    elif action == 4:
        point = point
    
    new_position = point[0] + point[1] * 5
    new_distance = ((point[0] - 4) ** 2 + (point[1] - 4) ** 2) ** 0.5
    reward = distance - new_distance
    terminated = new_distance == 0
    truncated = False

    client.data['position'] = new_position
    client.data['distance'] = new_distance

    res = (new_position, reward, terminated, truncated, {})
    return res

# distance is straight line distance from goal
distance = 32 ** 0.5

client = client.Client(action_space=5, state_space=25, func=generate_data, host='localhost', port=7234, distance=distance, position=0)
client.start()