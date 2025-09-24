# Q-Learning
Q-Learning and BFS (Breadth-First Search) solve maze.

```
# Example Maze
maze = [
    ['S', '.', '#', '#', '#'],
    ['.', '.', '.', '#', 'E'],
    ['#', '#', '.', '#', '.'],
    ['#', '.', '.', '.', '.'],
    ['#', '#', '#', '#', '#']
]
```

Where `S` is the start point, `E` is the end point, `.` is a passable path, and `#` is a wall.

- state: represented as coordinates in the maze, e.g., (0, 0) is the top-left corner of the maze.
- action: represented as the four possible movements (up, down, left, right).
- reward: the reward received after taking an action. Reaching the end point gives a reward of 10, hitting a wall or going out of bounds gives -1, and all other steps give -0.1 (to encourage reaching the end quickly).
- policy: $\epsilon$-greedy strategy.
- $ Q(s, a) $ update rule: 
$$ Q(s, a) \leftarrow Q(s, a) + \alpha\left[r' + \gamma \max_{a' \in \mathcal{A}} Q(s', a') - Q(s, a)\right] $$

Only numpy is required to run the code.
To run the Q-Learning and show the learned path:
```bash
python q.py
```
To run the BFS (Breadth-First Search) and show the optimal path:
```bash
python bfs.py
```