import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, LineString, Point
import heapq
import time

# Define grid size and create grid
GRID_SIZE = 25

# Define different wall configurations
WALLS = [        
    Polygon([(0, 0), (GRID_SIZE, 0), (GRID_SIZE, 1), (0, 1)]),
    Polygon([(0, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE), (0, GRID_SIZE)]),
    Polygon([(0, 0), (1, 0), (1, GRID_SIZE), (0, GRID_SIZE)]),
    Polygon([(GRID_SIZE - 1, 0), (GRID_SIZE, 0), (GRID_SIZE, GRID_SIZE), (GRID_SIZE - 1, GRID_SIZE)])
    ]

OBS_CONFIGS = {
    "simple": [
    ],
    "obstacles": [
        Polygon([(10, 10), (15, 10), (15, 15), (10, 15)]),
        Polygon([(5, 1), (10, 1), (10, 6), (5, 6)])
    ],
    "maze": [
        Polygon([(5, 5), (10, 5), (10, 10), (5, 10)]),
        Polygon([(15, 5), (20, 5), (20, 10), (15, 10)]),
        Polygon([(5, 15), (10, 15), (10, 20), (5, 20)]),
        Polygon([(17, 1), (18, 1), (18, 5), (17, 5)]),
        Polygon([(20, 17), (24, 17), (24, 18), (20, 18)]),        
        Polygon([(15, 15), (20, 15), (20, 20), (15, 20)])
    ]
}

# -------------------------------
# Helper Functions
# -------------------------------
def polygons_to_grid(walls=WALLS, grid_size=GRID_SIZE, obstacles=[]):
    """
    Converts polygons (walls and obstacles) into a numpy grid.
    """
    grid = np.zeros((grid_size, grid_size), dtype=int)

    def mark_polygon(polygon, grid):
        for x in range(grid_size):
            for y in range(grid_size):
                point = Point(x + 0.5, y + 0.5)
                if polygon.contains(point):
                    grid[x, y] = 1

    for wall in walls:
        mark_polygon(wall, grid)

    for obstacle in obstacles:
        mark_polygon(obstacle, grid)

    return grid

# -------------------------------
# Node Class
# -------------------------------
class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.g = float('inf')
        self.rhs = float('inf')
        self.key = (float('inf'), float('inf'))
        self.parent = None
        self.c = 1

    def __lt__(self, other):
        return self.key < other.key

class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.item_dict = {}
        self.REMOVED = "<removed>"

    def empty(self):
        return len(self.elements) == 0

    def insert(self, item, priority):
        if item in self.item_dict:
            self.remove(item)
        entry = (priority, item)
        self.item_dict[item] = entry
        heapq.heappush(self.elements, entry)
        print(f"Inserted item at ({item.x}, {item.y}) with priority {priority}")

    def top(self):
        while self.elements:
            priority, item = heapq.heappop(self.elements)
            if item is not self.REMOVED:
                del self.item_dict[item]
                print(f"Popped item at ({item.x}, {item.y}) with priority {priority}")
                return item
        raise KeyError("Tried to pop from empty priority queue")

    def top_key(self):
        if self.empty():
            return (float('inf'), float('inf'))
        return self.elements[0][0]

    def remove(self, item):
        if item in self.item_dict:
            entry = self.item_dict.pop(item)
            entry[-1] = self.REMOVED
            print(f"Removed item at ({item.x}, {item.y})")

# -------------------------------
# DStarLite Class
# -------------------------------
class DStarLite:
    def __init__(self, start_pos, goal_pos, walls, obs):
        self.walls = walls
        self.obstacles = obs
        self.goal_pos = goal_pos
        self.goal = None
        self.start_pos = start_pos
        self.start = None
        self.U = PriorityQueue()   
        self.k_m = 0
        self.nodes = []
        self.known_grid = polygons_to_grid()
        self.actual_grid = polygons_to_grid(WALLS, GRID_SIZE, self.obstacles)
        self.initialize()

    def initialize(self):    
        # Initialize all nodes
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                s = Node(x, y)
                if s.x == self.goal_pos[0] and s.y == self.goal_pos[1]:
                    s.rhs = 0
                    self.goal = s  # Assign goal node
                    print(f"Goal node assigned at ({s.x}, {s.y})")
                if s.x == self.start_pos[0] and s.y == self.start_pos[1]:
                    self.start = s  # Assign start node
                    print(f"Start node assigned at ({s.x}, {s.y})")
                self.nodes.append(s)

        # Insert the goal node into the priority queue
        if self.goal:
            self.U.insert(self.goal, self.calculate_key(self.goal))
            print(f"Goal node inserted into priority queue with key {self.calculate_key(self.goal)}")

        # Initialize the start node
        if self.start:
            self.start.rhs = self.heuristic(self.start, self.goal)
            self.U.insert(self.start, self.calculate_key(self.start))
            print(f"Start node inserted into priority queue with key {self.calculate_key(self.start)}")

    def successors(self, s):
        x = s.x
        y = s.y
        return[Node(x-1, y, s), Node(x+1, y, s), Node(x, y-1, s), Node(x, y+1, s),
               Node(x-1, y-1, s), Node(x-1, y+1, s), Node(x+1, y-1, s), Node(x+1, y+1, s)]  
    
    def predecessors(self, s):
        x = s.x
        y = s.y
        predecessors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    predecessors.append(Node(nx, ny))
        return predecessors
    
    def compute_rhs(self, s):
        if s == self.start:
            return 0
        return min(self.heuristic(s, s_prime) + self.g[s_prime] for s_prime in self.successors(s))

    def scan_for_changes(self, u):
        changed_edges = {}
        for v in self.successors(u):
            c_old = self.cost(u, v)
            x, y = v.x, v.y
            if self.known_grid[x, y] != self.actual_grid[x, y]:
                self.known_grid[x, y] = self.actual_grid[x, y]
                changed_edges[(u, v)] = c_old
        return changed_edges

    def heuristic(self, node, goal):
        return max(abs(node.x - goal.x), abs(node.y - goal.y))
    
    def cost(self, s, s_prime):
        """
        Computes the cost of moving from node s to node s_prime.
        Uses the known_grid for path planning and the actual_grid for occupancy checks.
        """
        # Check if s_prime is out of bounds
        if s_prime.x < 0 or s_prime.x >= GRID_SIZE or s_prime.y < 0 or s_prime.y >= GRID_SIZE:
            return float('inf')  # Out of bounds

        # Check if s_prime is occupied in the actual grid
        if self.actual_grid[s_prime.x, s_prime.y] == 1:
            print(f"Node at ({s_prime.x}, {s_prime.y}) is occupied")
            return float('inf')  # Occupied

        # Check if the edge is blocked by walls or obstacles in the known grid
        if self.known_grid[s_prime.x, s_prime.y] == 1:
            print(f"Node at ({s_prime.x}, {s_prime.y}) is blocked in the known grid")
            return float('inf')  # Blocked

        # If none of the above, the edge is traversable
        print(f"Edge from ({s.x}, {s.y}) to ({s_prime.x}, {s_prime.y}) is traversable")
        return 1

    def calculate_key(self, s):
        """
        Calculate the key for a node.
        """
        k1 = min(s.g, s.rhs) + self.heuristic(self.start, s) + self.k_m
        k2 = min(s.g, s.rhs)
        return (k1, k2)  # Return a tuple instead of a list

    def update_vertex(self, u):
        if u.g != u.rhs and u in self.U.item_dict:
            key = self.calculate_key(u)
            self.U.insert(u, key)
        elif u.g != u.rhs and u not in self.U.item_dict:
            key = self.calculate_key(u)
            self.U.insert(u, key)
        elif u.g == u.rhs and u in self.U.item_dict:
            self.U.remove(u)

    def compute_shortest_path(self):
        while not self.U.empty() and (self.U.top_key() < self.calculate_key(self.start) or self.start.rhs > self.start.g):
            u = self.U.top()
            k_old = self.U.top_key()
            k_new = self.calculate_key(u)

            if k_old < k_new:
                self.U.insert(u, k_new)
            elif u.g > u.rhs:
                u.g = u.rhs
                self.U.remove(u)

                # Update predecessors
                for s in self.predecessors(u):
                    if s != self.goal:
                        s.rhs = min([self.cost(s, sp) + sp.g for sp in self.successors(s)])
                    self.update_vertex(s)
            else:
                g_old = u.g
                u.g = float('inf')

                # Update predecessors
                for s in self.predecessors(u):
                    if s.rhs == self.cost(s, u) + g_old:
                        if s != self.goal:
                            s.rhs = min([self.cost(s, sp) + sp.g for sp in self.successors(s)])
                    self.update_vertex(s)

    def plan(self):
        s_last = self.start
        self.compute_shortest_path()  # Compute the initial path

        while self.start != self.goal:
            if self.start.rhs == float('inf'):
                print('No known path... Returning failure')
                break

            # Move the robot to the best successor
            min_cost = float('inf')
            best_successor = None
            for s_prime in self.successors(self.start):
                cost = self.cost(self.start, s_prime) + s_prime.g
                print(f"Cost to move to ({s_prime.x}, {s_prime.y}): {cost}")
                if cost < min_cost:
                    min_cost = cost
                    best_successor = s_prime

            if best_successor is not None:
                print(f"Moving robot to ({best_successor.x}, {best_successor.y})")
                self.start = best_successor
            else:
                print("No valid successor found. Path planning failed.")
                break

            # Simulate robot movement
            self.robot.move((self.start.x, self.start.y))

            # Scan for changes in the environment
            changed_edges = self.scan_for_changes(self.start)
            if changed_edges:
                self.k_m += self.heuristic(s_last, self.start)
                s_last = self.start

                # Update affected vertices
                for (u, v), c_old in changed_edges.items():
                    c_new = self.cost(u, v)
                    if c_old > c_new:
                        if u != self.goal:
                            u.rhs = min(u.rhs, self.cost(u, v) + v.g)
                    elif u.rhs == c_old + v.g:
                        if u != self.goal:
                            u.rhs = min([self.cost(u, sp) + sp.g for sp in self.successors(u)])
                    self.update_vertex(u)

                # Recompute the shortest path
                self.compute_shortest_path()

# -------------------------------
# Robot Class
# -------------------------------
class Robot:
    def __init__(self, pos):
        self.x, self.y = pos

    def move(self, newpoint):
        self.x, self.y = newpoint
    
    def get_pos(self):
        return (self.x, self.y)

# -------------------------------
# Main Class
# -------------------------------
class Main:
    def __init__(self, map_name="simple"):
        self.walls = WALLS
        self.obstacles = OBS_CONFIGS[map_name]
        self.start_pos = (2, 2)
        self.goal_pos = (GRID_SIZE - 3, GRID_SIZE - 3)
        self.robot = Robot(self.start_pos)
        self.dstar = DStarLite(self.start_pos, self.goal_pos, self.walls, self.obstacles)
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_aspect('equal', 'box')

    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_aspect('equal', 'box')
        
        # Plot Grid
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, color='blue', lw=0.5))

        # Plot walls
        for wall in self.walls:
            x, y = wall.exterior.xy
            self.ax.fill(x, y, alpha=0.5, fc='black', edgecolor='black')

        # Plot obstacles
        for obstacle in self.obstacles:
            x, y = obstacle.exterior.xy
            self.ax.fill(x, y, alpha=0.5, fc='yellow', edgecolor='yellow')

        # Plot start and goal
        self.ax.plot(self.start_pos[0]+.5, self.start_pos[1]+.5, 'go', markersize=10, label='Start')
        self.ax.plot(self.goal_pos[0]+.5, self.goal_pos[1]+.5, 'ro', markersize=10, label='Goal')

        # Plot robot
        self.ax.plot(self.robot.x+.5, self.robot.y+.5, 'bo', markersize=10, label='Robot')

        # Plot path
        path = self.dstar.compute_shortest_path()
        if path:
            path_x = [p.x+.5 for p in path]
            path_y = [p.y+.5 for p in path]
            self.ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')

        plt.legend()
        plt.draw()
        plt.pause(0.1)

    def run(self):
        while self.robot.x != self.goal_pos[0] or self.robot.y != self.goal_pos[1]:
            self.dstar.plan()
            self.update_plot()
            time.sleep(1)

if __name__ == "__main__":
    main = Main()
    main.run()