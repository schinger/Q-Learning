import numpy as np
import random

def solve_maze_q_learning(maze, learning_rate=0.1, discount_factor=0.9, epsilon_start=1.0, epsilon_decay_rate=0.001, num_episodes=1000, max_steps_per_episode=100):
    """
    Solves a maze using Q-Learning.

    Args:
        maze: A 2D list representing the maze. 'S' for start, 'E' for end, '#' for walls, and '.' for open paths.
        learning_rate (float): Learning rate (alpha).
        discount_factor (float): Discount factor (gamma).
        epsilon_start (float): Initial exploration rate (epsilon).
        epsilon_decay_rate (float): Decay rate for epsilon.
        num_episodes (int): Number of training episodes.
        max_steps_per_episode (int): Maximum steps per episode to prevent infinite loops.

    Returns:
        Q-table (dict): Learned Q-table.
        path (list): Path found using the learned Q-table (greedy policy), or None if no path found.
    """
    rows = len(maze)
    cols = len(maze[0])

    start_pos = None
    end_pos = None
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 'S':
                start_pos = (r, c)
            elif maze[r][c] == 'E':
                end_pos = (r, c)

    if not start_pos or not end_pos:
        return {}, None  # Start or end not found

    # Initialize Q-table: Q[state][action] = Q-value
    q_table = {}
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] != '#': # Walls are not states
                q_table[(r, c)] = [0.0] * 4  # 4 actions: Right, Left, Down, Up

    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Right, Left, Down, Up (actions indices 0, 1, 2, 3)

    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = start_pos
        for step in range(max_steps_per_episode):
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action_index = random.choice(range(len(actions))) # Explore
            else:
                action_index = np.argmax(q_table[state]) # Exploit

            action = actions[action_index]
            next_r, next_c = state[0] + action[0], state[1] + action[1]
            next_state = (next_r, next_c)

            # Calculate reward
            if not (0 <= next_r < rows and 0 <= next_c < cols): # Out of bounds
                reward = -1.0 # Punish going out of bounds
                next_state = state # Stay in the same state
            elif maze[next_r][next_c] == '#': # Hit a wall
                reward = -1.0
                next_state = state # Stay in the same state
            elif maze[next_r][next_c] == 'E': # Reached the goal
                reward = 10.0
            else: # Moved to an open path or 'S'
                reward = -0.1

            # Q-value update
            q_table[state][action_index] += learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state][action_index])

            state = next_state

            if maze[state[0]][state[1]] == 'E': # Reached goal, end episode
                break

        epsilon = max(epsilon_start * (1 - epsilon_decay_rate * episode), 0.01) # Decay epsilon, but keep a minimum exploration rate


    # Path finding using learned Q-table (Greedy policy)
    path = []
    current_state = start_pos
    path.append(current_state)
    for _ in range(max_steps_per_episode):
        if current_state not in q_table: # Should not happen in a well-trained Q-table, but for robustness
            return q_table, None
        best_action_index = np.argmax(q_table[current_state])
        best_action = actions[best_action_index]
        next_r, next_c = current_state[0] + best_action[0], current_state[1] + best_action[1]
        next_state = (next_r, next_c)

        if not (0 <= next_r < rows and 0 <= next_c < cols) or maze[next_r][next_c] == '#':
            break # Stop if going out of bounds or into a wall in exploitation

        path.append(next_state)
        current_state = next_state

        if maze[current_state[0]][current_state[1]] == 'E':
            return q_table, path

    return q_table, None # No path found


# Example Maze (same as before)
maze = [
    ['S', '.', '#', '#', '#'],
    ['.', '.', '.', '#', 'E'],
    ['#', '#', '.', '#', '.'],
    ['#', '.', '.', '.', '.'],
    ['#', '#', '#', '#', '#']
]

learned_q_table, q_learning_path = solve_maze_q_learning(maze)

if q_learning_path:
    print("Q-Learning Path found:", q_learning_path)
    maze_with_path_ql = [list(row) for row in maze]
    for r, c in q_learning_path:
        if maze_with_path_ql[r][c] not in ['S', 'E']:
            maze_with_path_ql[r][c] = 'X'
    for row in maze_with_path_ql:
        print("".join(row))

else:
    print("Q-Learning: No path found.")