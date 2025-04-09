def solve_maze_bfs(maze):
    """
    Solves a maze using Breadth-First Search (BFS).

    Args:
        maze: A 2D list representing the maze. 'S' for start, 'E' for end,
              '#' for walls, and '.' for open paths.

    Returns:
        A list of coordinates representing the shortest path from start to end,
        or None if no path exists.
    """
    start_pos = None
    end_pos = None
    rows = len(maze)
    cols = len(maze[0])

    # Find start and end positions
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 'S':
                start_pos = (r, c)
            elif maze[r][c] == 'E':
                end_pos = (r, c)

    if not start_pos or not end_pos:
        return None  # Start or end position not found

    queue = [start_pos] # Initialize queue with start position
    visited = set([start_pos])  # Keep track of visited cells
    parent = {}  # Store parent of each cell for path reconstruction

    while queue:
        current_pos = queue.pop(0)
        if current_pos == end_pos:
            return reconstruct_path(parent, end_pos, start_pos)

        r, c = current_pos
        # Explore neighbors 
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_r, neighbor_c = r + dr, c + dc
            neighbor_pos = (neighbor_r, neighbor_c)

            if (0 <= neighbor_r < rows and 0 <= neighbor_c < cols and  # Check boundaries
                maze[neighbor_r][neighbor_c] != '#' and  # Check for wall
                neighbor_pos not in visited):  # Check if visited
                visited.add(neighbor_pos)
                parent[neighbor_pos] = current_pos  # Set parent for path reconstruction
                queue.append(neighbor_pos)

    return None  # No path found


def reconstruct_path(parent, end_pos, start_pos):
    """
    Reconstructs the path from the end position back to the start position
    using the parent dictionary.
    """
    path = []
    current_pos = end_pos
    while current_pos != start_pos:
        path.append(current_pos)
        current_pos = parent[current_pos]
    path.append(start_pos)
    return path[::-1]  # Reverse the path to get start to end


# Example Maze
maze = [
    ['S', '.', '#', '#', '#'],
    ['.', '.', '.', '#', 'E'],
    ['#', '#', '.', '#', '.'],
    ['#', '.', '.', '.', '.'],
    ['#', '#', '#', '#', '#']
]

path = solve_maze_bfs(maze)

if path:
    print("Path found:", path)
    # Visualize the path on the maze (optional)
    maze_with_path = [list(row) for row in maze] # Create a mutable copy
    for r, c in path:
        if maze_with_path[r][c] not in ['S', 'E']:
            maze_with_path[r][c] = 'X' # Mark path with 'X'
    for row in maze_with_path:
        print("".join(row))

else:
    print("No path found.")