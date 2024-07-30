import numpy as np
import random
import sys
import heapq
import math

WINDOW_SIZE =  (1160, 760)
CELL_SIZE = (40, 40)
MAP_SIZE = (int(round(WINDOW_SIZE[0]/CELL_SIZE[0])),int(round(WINDOW_SIZE[1]/CELL_SIZE[1])))


# Define helper functions
def heuristic(start, target):
    """
    Calculate the heuristic (Manhattan distance) between two points.
    """
    return abs(target[0] - start[0]) + abs(target[1] - start[1])

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
    Get neighboring positions (up, down, left, right, and diagonal).
    """
    neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y),   # Up, down, left, right
                 (x - 1, y - 1), (x + 1, y - 1),                 # Diagonal top-left, top-right
                 (x - 1, y + 1), (x + 1, y + 1)]                 # Diagonal bottom-left, bottom-right
    return neighbors

def reconstruct_path(came_from, current):
    """
    Reconstruct the path from start to current position.
    """
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    return path[::-1]  # Reverse the path

def astar(start, target, avoid_blocks):
    """
    A* algorithm to find the shortest path from start to target.
    """
    open_set = []
    closed_set = set()
    heapq.heappush(open_set, (0, start))
    g_score = {start: 0}
    f_score = {start: heuristic(start, target)}
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target:
            return reconstruct_path(came_from, current)

        closed_set.add(current)

        for neighbor in get_neighbors(*current):
            if not is_valid_move(*neighbor, avoid_blocks):
                continue

            tentative_g_score = g_score[current] + math.sqrt((current[0] - neighbor[0]) ** 2 + (current[1] - neighbor[1]) ** 2)

            if neighbor in closed_set and tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, target)
                if neighbor not in closed_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

def find_nearest_valid_point(target, avoid_blocks):
    """
    Find the nearest valid point to the target position.
    """
    valid_points = [(x, y) for x in range(MAP_SIZE[0]) for y in range(MAP_SIZE[1]) if (x, y) not in avoid_blocks]
    if valid_points:
        nearest_point = min(valid_points, key=lambda p: heuristic(p, target))
        return nearest_point
    return None

def astar_with_avoid_blocks(start, target, avoid_blocks):
    """
    A* algorithm to find the shortest path from start to target while avoiding blocks.
    """
    if target in avoid_blocks:
        target = find_nearest_valid_point(target, avoid_blocks)
        if target is None:
            return None
    if start in avoid_blocks:   ############### Added my self
        start = find_nearest_valid_point(start, avoid_blocks)
        if start is None:
            return None

    return astar(start, target, avoid_blocks)

# Example usage
start_position = (22, 18)  # Bot's starting position
target_position = (2, 2)  # Target position
avoid_blocks = [(8, 0), (19, 0), (0, 5), (11, 5), (0, 14), (4, 2), (22, 10), (3, 6), (15, 7), (7, 3), (18, 3), (7, 12), (18, 12), (0, 0), (11, 0), (22, 5), (3, 1), (14, 1), (3, 10), (14, 10), (14, 19), (15, 2), (15, 11), (7, 7), (18, 7), (10, 8), (10, 17), (25, 15), (22, 0), (3, 5), (14, 5), (15, 6), (7, 2), (18, 2), (7, 11), (18, 11), (10, 3), (10, 12), (2, 8), (3, 0), (14, 0), (14, 18), (6, 10), (21, 8), (-1, 15), (10, 7), (2, 3), (2, 12), (25, 14), (22, 8), (17, 6), (9, 11), (6, 5), (21, 3), (21, 12), (10, 2), (10, 11), (2, 7), (17, 1), (5, 8), (17, 10), (9, 6), (6, 0), (21, 7), (-1, 14), (10, 6), (2, 2), (2, 11), (13, 8), (24, 8), (13, 17), (24, 17), (5, 3), (17, 5), (9, 1), (5, 12), (9, 10), (9, 19), (21, 2), (13, 3), (24, 3), (1, 10), (13, 12), (24, 12), (16, 8), (17, 0), (5, 7), (9, 5), (20, 1), (20, 10), (12, 6), (23, 6), (4, 11), (1, 5), (13, 7), (24, 7), (16, 3), (24, 16), (16, 12), (5, 2), (9, 0), (5, 11), (11, 18), (20, 5), (12, 1), (23, 1), (12, 10), (23, 10), (4, 6), (12, 19), (1, 0), (13, 2), (24, 2), (13, 11), (24, 11), (16, 7), (5, 6), (8, 8), (19, 8), (20, 0), (12, 5), (23, 5), (4, 1), (4, 10), (24, 6), (16, 2), (8, 3), (19, 3), (8, 12), (19, 12), (0, 8), (11, 8), (11, 17), (0, 17), (12, 0), (23, 0), (4, 5), (15, 1), (15, 10), (7, 6), (18, 6), (15, 19), (8, 7), (19, 7), (0, 3), (11, 3), (0, 12), (11, 12), (4, 0), (15, 5), (7, 1), (18, 1), (7, 10), (18, 10), (8, 2), (19, 2), (8, 11), (19, 11), (0, 7), (11, 7), (0, 16), (22, 3), (22, 12), (3, 8), (14, 8), (14, 17), (15, 0), (7, 5), (18, 5), (15, 18), (19, 6), (0, 2), (11, 2), (22, 7), (3, 3), (14, 3), (3, 12), (14, 12), (7, 0), (18, 0), (21, 11), (10, 1), (10, 10), (2, 6), (10, 19), (7, 8), (25, 17), (22, 2), (22, 11), (3, 7), (14, 7), (6, 8), (21, 6), (10, 5), (2, 1), (2, 10), (22, 6), (3, 2), (14, 2), (3, 11), (14, 11), (9, 18), (6, 3), (21, 1), (6, 12), (21, 10), (-1, 17), (10, 0), (2, 5), (10, 18), (25, 16), (22, 1), (14, 6), (17, 8), (6, 7), (21, 5), (2, 0), (13, 6), (24, 15), (16, 11), (5, 1), (17, 3), (5, 10), (17, 12), (9, 8), (9, 17), (6, 2), (21, 0), (6, 11), (-1, 16), (12, 18), (13, 1), (24, 1), (1, 8), (13, 10), (24, 10), (16, 6), (13, 19), (5, 5), (17, 7), (9, 3), (9, 12), (6, 6), (20, 8), (1, 3), (13, 5), (24, 5), (16, 1), (1, 12), (24, 14), (16, 10), (5, 0), (17, 2), (17, 11), (9, 7), (6, 1), (20, 3), (20, 12), (12, 8), (23, 8), (12, 17), (13, 0), (24, 0), (1, 7), (16, 5), (13, 18), (9, 2), (8, 6), (0, 11), (11, 11), (20, 7), (12, 3), (23, 3), (12, 12), (23, 12), (4, 8), (1, 2), (16, 0), (1, 11), (8, 1), (19, 1), (8, 10), (19, 10), (0, 6), (11, 6), (0, 15), (20, 2), (20, 11), (12, 7), (23, 7), (4, 3), (4, 12), (1, 6), (15, 8), (15, 17), (8, 5), (19, 5), (0, 1), (11, 1), (0, 10), (11, 10), (11, 19), (20, 6), (12, 2), (23, 2), (12, 11), (23, 11), (4, 7), (1, 1), (15, 3), (15, 12), (18, 8)]
path = astar_with_avoid_blocks(start_position, target_position, avoid_blocks)
if path:
    print("Path found:", path)
else:
    print("No path found to the target while avoiding obstacles.")
