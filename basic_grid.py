import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
import heapq
import itertools
import time

# -------------------------------
# Global Definitions
# -------------------------------
GRID_SIZE = 25  # Size of the grid (25x25)
GOAL_POS = (23, 23)  # Goal position on the grid
START_POS = (2, 2)  # Starting position on the grid
FPS = 10  # Frames per second for visualization
FRAME_DELAY = 1 / FPS  # Delay between frames for visualization

# Define the walls of the grid as polygons
WALLS = [
    Polygon([(0, 0), (GRID_SIZE, 0), (GRID_SIZE, 1), (0, 1)]),  # Bottom wall
    Polygon([(0, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE), (0, GRID_SIZE)]),  # Top wall
    Polygon([(0, 0), (1, 0), (1, GRID_SIZE), (0, GRID_SIZE)]),  # Left wall
    Polygon([(GRID_SIZE - 1, 0), (GRID_SIZE, 0), (GRID_SIZE, GRID_SIZE), (GRID_SIZE - 1, GRID_SIZE)])  # Right wall
]

# Obstacle configurations for different maps
OBS_CONFIGS = {
    "maze": [
        Polygon([(5, 5), (10, 5), (10, 10), (5, 10)]),
        Polygon([(15, 5), (20, 5), (20, 10), (15, 10)]),
        Polygon([(5, 15), (10, 15), (10, 20), (5, 20)]),
        Polygon([(17, 1), (18, 1), (18, 5), (17, 5)]),
        Polygon([(20, 17), (24, 17), (24, 18), (20, 18)]),
        Polygon([(15, 15), (20, 15), (20, 20), (15, 20)])
    ],
    "simple": [],  # No obstacles
    "obstacles": [
        Polygon([(10, 10), (15, 10), (15, 15), (10, 15)]),
        Polygon([(5, 1), (10, 1), (10, 6), (5, 6)])
    ]
}

def polygons_to_grid(walls=WALLS, grid_size=GRID_SIZE, obstacles=[]):
    """
    Convert polygons (walls and obstacles) to a grid representation.
    
    Args:
        walls (list): List of wall polygons.
        grid_size (int): Size of the grid.
        obstacles (list): List of obstacle polygons.
    
    Returns:
        np.ndarray: A grid where 1 represents blocked cells (walls/obstacles) and 0 represents free cells.
    """
    grid = np.zeros((grid_size, grid_size), dtype=int)
    def mark_polygon(polygon, grid):
        """
        Mark cells in the grid that are inside the given polygon as blocked (1).
        
        Args:
            polygon (Polygon): The polygon to mark on the grid.
            grid (np.ndarray): The grid to mark the polygon on.
        """
        for x in range(grid_size):
            for y in range(grid_size):
                pt = Point(x + 0.5, y + 0.5)
                if polygon.contains(pt):
                    grid[x, y] = 1
    for wall in walls:
        mark_polygon(wall, grid)
    for obs in obstacles:
        mark_polygon(obs, grid)
    return grid

# -------------------------------
# Node Class
# -------------------------------
class Node:
    """
    Represents a node in the grid for path planning.
    
    Attributes:
        x (int): X-coordinate of the node.
        y (int): Y-coordinate of the node.
        g (float): Cost from start to this node.
        rhs (float): Right Hand Side value for D* Lite algorithm.
        key (tuple): Key for priority queue sorting.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = float('inf')
        self.rhs = float('inf')
        self.key = (float('inf'), float('inf'))
    def __lt__(self, other):
        return self.key < other.key
    def __hash__(self):
        return hash((self.x, self.y))
    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)
    def __repr__(self):
        return f"Node({self.x}, {self.y})"

# -------------------------------
# Priority Queue Class
# -------------------------------
class PriorityQueue:
    """
    A priority queue implementation for managing nodes in the D* Lite algorithm.
    
    Attributes:
        pq (list): The priority queue list.
        entry_finder (dict): A dictionary to find entries in the priority queue.
        REMOVED (str): Marker for removed tasks.
        counter (itertools.count): A counter for unique task IDs.
    """
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()
    def empty(self):
        return not self.pq
    def insert(self, item, priority):
        if item in self.entry_finder:
            self.remove(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)
    def remove(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED
    def top(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item
        raise KeyError("pop from an empty priority queue")
    def top_key(self):
        while self.pq:
            priority, count, item = self.pq[0]
            if item is self.REMOVED:
                heapq.heappop(self.pq)
                continue
            return priority
        return (float('inf'), float('inf'))
    def update(self, item, priority):
        self.insert(item, priority)

# -------------------------------
# DStarLite Class
# -------------------------------
class DStarLite:
    """
    Implements the D* Lite algorithm for dynamic path planning.
    
    Attributes:
        walls (list): List of wall polygons.
        obstacles (list): List of obstacle polygons.
        goal_pos (tuple): Goal position on the grid.
        start_pos (tuple): Starting position on the grid.
        k_m (float): Key modifier for D* Lite.
        U (PriorityQueue): Priority queue for managing nodes.
        known_grid (np.ndarray): Grid representing known obstacles.
        actual_grid (np.ndarray): Grid representing actual obstacles.
        nodes (dict): Dictionary of nodes in the grid.
        start (Node): Start node.
        goal (Node): Goal node.
        robot (Robot): The robot moving on the grid.
        visualizer (Main): Visualizer for the grid and path.
        original_path (list): The original planned path.
        actual_path (list): The actual path traveled by the robot.
    """
    def __init__(self, start_pos, goal_pos, walls, obs, visualizer=None):
        self.walls = walls
        self.obstacles = obs
        self.goal_pos = goal_pos
        self.start_pos = start_pos
        self.k_m = 0
        self.U = PriorityQueue()
        self.known_grid = polygons_to_grid(walls)
        self.actual_grid = polygons_to_grid(walls, GRID_SIZE, self.obstacles)
        self.nodes = {}
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.nodes[(x, y)] = Node(x, y)
        self.start = self.nodes[self.start_pos]
        self.goal = self.nodes[self.goal_pos]
        self.goal.rhs = 0
        self.U.insert(self.goal, self.calculate_key(self.goal))
        self.robot = Robot(start_pos)
        self.visualizer = visualizer
        self.original_path = None
        self.actual_path = []  # Track the actual path traveled

    def plan(self):
        """
        Plan the path from start to goal using the D* Lite algorithm.
        """
        if self.known_grid[self.start.x, self.start.y] == 1:
            print("Start cell is blocked. No valid path.")
            return
        s_last = self.start
        self.compute_shortest_path()
        if self.original_path is None:
            self.original_path = self.get_path()  # Store the original planned path
        current_path = self.get_path()
        if self.visualizer:
            self.visualizer.update_plot(self.robot,
                                       original_path=self.original_path,
                                       current_path=current_path,
                                       known_grid=self.known_grid)
        while self.start != self.goal:
            if self.start.rhs == float('inf'):
                print("No known path... Returning failure")
                return
            best_successor = None
            min_cost = float('inf')
            for s in self.successors(self.start):
                cost_val = self.cost(self.start, s) + s.g
                if cost_val < min_cost:
                    min_cost = cost_val
                    best_successor = s
            if best_successor is None:
                print("No valid successor found. Path planning failed.")
                return
            self.start = best_successor
            self.robot.move(self.start)
            self.actual_path.append(self.start)  # Add the current position to the actual path
            current_path = self.get_path()
            if self.visualizer:
                self.visualizer.update_plot(self.robot,
                                            original_path=self.original_path,
                                            current_path=current_path,
                                            known_grid=self.known_grid)
            time.sleep(FRAME_DELAY)
            changed_vertices = self.scan_for_changes(self.start)
            if changed_vertices:
                self.k_m += self.heuristic(s_last, self.start)
                s_last = self.start
                for vertex in changed_vertices:
                    self.update_vertex(vertex)
                self.compute_shortest_path()
            print(f"Robot moved to: ({self.robot.x}, {self.robot.y})")
        print("Goal reached!")
        if self.visualizer:
            self.visualizer.update_plot(self.robot,
                                       original_path=self.original_path,
                                       actual_path=self.actual_path,  # Pass the actual path
                                       known_grid=self.known_grid)
            time.sleep(2)  # Pause to show the final plot before moving to the next map

    def heuristic(self, a, b):
        """
        Calculate the heuristic distance between two nodes.
        
        Args:
            a (Node): The first node.
            b (Node): The second node.
        
        Returns:
            float: The heuristic distance between the two nodes.
        """
        return max(abs(a.x - b.x), abs(a.y - b.y))
    
    def successors(self, s):
        """
        Get the successors of a node.
        
        Args:
            s (Node): The node to get successors for.
        
        Returns:
            list: List of successor nodes.
        """
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            nx, ny = s.x + dx, s.y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                neighbors.append(self.nodes[(nx, ny)])
        return neighbors
    
    def predecessors(self, s):
        """
        Get the predecessors of a node.
        
        Args:
            s (Node): The node to get predecessors for.
        
        Returns:
            list: List of predecessor nodes.
        """
        return self.successors(s)
    
    def cost(self, s, s_prime):
        """
        Calculate the cost between two nodes.
        
        Args:
            s (Node): The first node.
            s_prime (Node): The second node.
        
        Returns:
            float: The cost between the two nodes.
        """
        if self.known_grid[s_prime.x, s_prime.y] == 1:
            return float('inf')
        return 1
    
    def calculate_key(self, s):
        """
        Calculate the key for a node.
        
        Args:
            s (Node): The node to calculate the key for.
        
        Returns:
            tuple: The calculated key.
        """
        k1 = min(s.g, s.rhs) + self.heuristic(self.start, s) + self.k_m
        k2 = min(s.g, s.rhs)
        s.key = (k1, k2)
        return s.key
    
    def update_vertex(self, u):
        """
        Update the vertex in the priority queue.
        
        Args:
            u (Node): The node to update.
        """
        if u != self.goal:
            u.rhs = min([self.cost(s, u) + s.g for s in self.predecessors(u)] + [float('inf')])
        if u in self.U.entry_finder:
            self.U.remove(u)
        if u.g != u.rhs:
            self.U.insert(u, self.calculate_key(u))
    
    def compute_shortest_path(self):
        """
        Compute the shortest path using the D* Lite algorithm.
        """
        iterations = 0
        max_iterations = 10000
        while ((self.U.top_key() < self.calculate_key(self.start)) or (self.start.rhs != self.start.g)):
            iterations += 1
            if iterations > max_iterations:
                print("compute_shortest_path: iteration limit reached")
                break
            try:
                u = self.U.top()
            except KeyError:
                break
            k_old = u.key
            k_new = self.calculate_key(u)
            if k_old < k_new:
                self.U.insert(u, k_new)
            elif u.g > u.rhs:
                u.g = u.rhs
                for s in self.predecessors(u):
                    self.update_vertex(s)
            else:
                old_g = u.g
                u.g = float('inf')
                for s in self.predecessors(u) + [u]:
                    if s.rhs == self.cost(s, u) + old_g:
                        if s != self.goal:
                            s.rhs = min([self.cost(sp, s) + sp.g for sp in self.predecessors(s)] + [float('inf')])
                    self.update_vertex(s)
    
    def get_path(self):
        """
        Get the current planned path from start to goal.
        
        Returns:
            list: List of nodes representing the path.
        """
        path = []
        current = self.start
        path.append(current)
        while current != self.goal:
            best_cost = float('inf')
            best_node = None
            for s in self.successors(current):
                cost_val = self.cost(current, s) + s.g
                if cost_val < best_cost:
                    best_cost = cost_val
                    best_node = s
            if best_node is None or best_cost == float('inf'):
                break
            current = best_node
            path.append(current)
        return path

    def scan_for_changes(self, current):
        """
        Scan for changes in the grid (e.g., newly discovered obstacles).
        
        Args:
            current (Node): The current node of the robot.
        
        Returns:
            set: Set of nodes that have changed.
        """
        changed_vertices = set()
        for neighbor in self.successors(current):
            if self.known_grid[neighbor.x, neighbor.y] == 0 and self.actual_grid[neighbor.x, neighbor.y] == 1:
                self.known_grid[neighbor.x, neighbor.y] = 1
                changed_vertices.add(self.nodes[(neighbor.x, neighbor.y)])
        return changed_vertices


# -------------------------------
# Robot Class
# -------------------------------
class Robot:
    """
    Represents the robot moving on the grid.
    
    Attributes:
        x (int): X-coordinate of the robot.
        y (int): Y-coordinate of the robot.
    """
    def __init__(self, pos):
        self.x, self.y = pos
    def move(self, new_node):
        """
        Move the robot to a new node.
        
        Args:
            new_node (Node): The node to move the robot to.
        """
        self.x, self.y = new_node.x, new_node.y

# -------------------------------
# Main Visualization Class
# -------------------------------
class Main:
    """
    Handles the visualization of the grid, robot, and paths.
    
    Attributes:
        walls (list): List of wall polygons.
        obs (list): List of obstacle polygons.
        goal_pos (tuple): Goal position on the grid.
        start_pos (tuple): Starting position on the grid.
        wall_grid (np.ndarray): Grid representing walls.
        ax (matplotlib.axes.Axes): The axes for plotting.
    """
    def __init__(self, map_name="maze", ax=None):
        self.walls = WALLS
        self.obs = OBS_CONFIGS.get(map_name, OBS_CONFIGS["maze"])
        self.goal_pos = GOAL_POS
        self.start_pos = START_POS
        self.wall_grid = polygons_to_grid(self.walls)
        self.ax = ax
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_aspect('equal')
        self.update_plot(Robot(self.start_pos))
        
    def update_plot(self, robot, original_path=None, current_path=None, known_grid=None, actual_path=None):
        """
        Update the plot with the current state of the grid, robot, and paths.
        
        Args:
            robot (Robot): The robot to plot.
            original_path (list): The original planned path.
            current_path (list): The current planned path.
            known_grid (np.ndarray): The grid representing known obstacles.
            actual_path (list): The actual path traveled by the robot.
        """
        self.ax.clear()
        # Draw grid lines.
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, edgecolor='blue', lw=0.5))
        # Draw walls.
        for wall in self.walls:
            xw, yw = wall.exterior.xy
            self.ax.fill(xw, yw, color='black')
        # Draw obstacles in original (light gray) color.
        for obs in self.obs:
            xo, yo = obs.exterior.xy
            self.ax.fill(xo, yo, color='gray', alpha=0.5)
        # Overlay discovered obstacle cells (touched by the robot) as black.
        if known_grid is not None:
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    if known_grid[x, y] == 1 and self.wall_grid[x, y] == 0:
                        self.ax.add_patch(plt.Rectangle((x, y), 1, 1, color='black'))
        # Draw the goal.
        self.ax.add_patch(plt.Rectangle((self.goal_pos[0], self.goal_pos[1]), 1, 1, color='red'))
        # Draw the original path.
        if original_path is not None and len(original_path) > 1:
            xs = [node.x + 0.5 for node in original_path]
            ys = [node.y + 0.5 for node in original_path]
            self.ax.plot(xs, ys, linestyle='--', color='red', label="Estimated Path")
        # Draw the current path.
        if current_path is not None and len(current_path) > 1:
            xs = [node.x + 0.5 for node in current_path]
            ys = [node.y + 0.5 for node in current_path]
            self.ax.plot(xs, ys, linestyle='-', color='blue', label="Current Path")
        # Draw the actual path traveled.
        if actual_path is not None and len(actual_path) > 1:
            xs = [node.x + 0.5 for node in actual_path]
            ys = [node.y + 0.5 for node in actual_path]
            self.ax.plot(xs, ys, linestyle='-', color='green', linewidth=2, label="Actual Path")
        # Draw the robot.
        self.ax.add_patch(plt.Circle((robot.x + 0.5, robot.y + 0.5), 0.3, color='green'))
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_aspect('equal')
        # Only add a legend if there are handles with nonempty labels.
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc="upper right")
        plt.draw()
        plt.pause(FRAME_DELAY)

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    plt.ion()
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    map_configs = ["maze", "obstacles", "simple"]

    for map_name, ax in zip(map_configs, axs):
        visualizer = Main(map_name, ax)
        dstar = DStarLite(START_POS, GOAL_POS, WALLS, OBS_CONFIGS[map_name], visualizer=visualizer)
        dstar.plan()

    plt.ioff()
    plt.show()