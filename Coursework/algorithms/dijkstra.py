import heapq

WINDOW_SIZE = (1160, 760)
BOT_SIZE = (40, 40)
MAP_SIZE = (int(round(WINDOW_SIZE[0] / BOT_SIZE[0])), int(round(WINDOW_SIZE[1] / BOT_SIZE[1])))


def is_valid_position(x, y):
    """
    Check if a position is valid within the window boundaries.
    """
    return 0 <= x < MAP_SIZE[0] and 0 <= y < MAP_SIZE[1]


def is_valid_move(x, y, avoid_blocks):
    """
    Check if a move to the position (x, y) is valid, considering avoid_blocks.
    """
    return is_valid_position(x, y) and (x, y) not in avoid_blocks


def get_neighbors(x, y):
    """
    Get neighboring positions (up, down, left, right, diagonals).
    """
    neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
    diagonal_neighbors = [(x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)]
    return [neighbor for neighbor in neighbors + diagonal_neighbors if is_valid_position(*neighbor)]

def getScore(start, target):
    """
    Calculate the heuristic (Manhattan distance) between two points.
    """
    return abs(target[0] - start[0]) + abs(target[1] - start[1])

def dijkstra(start, target, avoid_blocks):
    """
    Dijkstra's algorithm to find the shortest path from start to target.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    g_score = {start: 0}
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Reverse the path

        for neighbor in get_neighbors(*current):
            if not is_valid_move(*neighbor, avoid_blocks):
                continue

            tentative_g_score = g_score[current] + getScore(current, neighbor)

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (tentative_g_score, neighbor))

    return None  # No path found


# Example usage
start_position = (22, 18)  # Bot's starting position
target_position = (2, 2)  # Target position
avoid_blocks = [(8, 0), (19, 0), (0, 5), (11, 5), (0, 14), (4, 2)]
path = dijkstra(start_position, target_position, avoid_blocks)
if path:
    print("Path found:", path)
else:
    print("No path found to the target while avoiding obstacles.")

