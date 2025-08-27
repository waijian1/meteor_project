class Platform:
    def __init__(self, id, x1, x2, y):
        self.id = id
        self.x1 = x1
        self.x2 = x2
        self.y = y

class Rope:
    def __init__(self, id, x, y_top, y_bottom):
        self.id = id
        self.x = x
        self.y_top = y_top
        self.y_bottom = y_bottom

class Node:
    def __init__(self, id, x, y, node_type="platform"):
        self.id = id
        self.x = x
        self.y = y
        self.type = node_type
        self.edges = []

class MapGraph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_edge(self, id1, id2, action, cost=1):
        self.nodes[id1].edges.append((id2, action, cost))

from collections import deque

def bfs_path(graph, start_id, goal_id):
    queue = deque([(start_id, [])])
    visited = set()

    while queue:
        current, path = queue.popleft()
        if current == goal_id:
            return path  # sequence of actions

        if current in visited:
            continue
        visited.add(current)

        for neighbor, action, cost in graph.nodes[current].edges:
            queue.append((neighbor, path + [action]))
    return None
