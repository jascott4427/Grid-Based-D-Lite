import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, LineString, Point
import heapq

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

    Inputs:
        - walls: List of polygons representing walls.
        - obstacles: List of polygons representing obstacles.
        - grid_size: Size of the grid (grid_size x grid_size).

    Outputs:
        - grid: A numpy array of size (grid_size x grid_size) where:
            - 0 represents free space.
            - 1 represents occupied space (walls or obstacles).
    """
    # Initialize an empty grid
    grid = np.zeros((grid_size, grid_size), dtype=int)

    # Function to mark cells inside a polygon as occupied
    def mark_polygon(polygon, grid):
        for x in range(grid_size):
            for y in range(grid_size):
                # Create a point at the center of the cell
                point = Point(x + 0.5, y + 0.5)
                # Check if the point is inside the polygon
                if polygon.contains(point):
                    grid[x, y] = 1

    # Mark walls on the grid
    for wall in walls:
        mark_polygon(wall, grid)

    # Mark obstacles on the grid
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
        self.g = float('inf')  # Cost from start to this node
        self.rhs = float('inf')  # One-step lookahead cost
        self.key = (float('inf'), float('inf'))  # Priority queue key
        self.parent = None
        self.c = 1

    def __lt__(self, other):
        return self.key < other.key

# -------------------------------
# Priority Queue
# -------------------------------
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

    def top(self):
        if not self.empty:
            priority, item = heapq.heappop(self.elements)
            if item is not self.REMOVED:
                del self.item_dict[item]
                return item
        raise KeyError("Tried to pop from empty priority queue")

    def top_key(self):
        if self.empty():
            return (float('inf'), float('inf'))
        return self.elements[0][0]

    def remove(self, item):
        del_item = self.item_dict.pop(item)
        del_item[-1] = self.REMOVED
            

# -------------------------------
# DStarLite Class based on https://idm-lab.org/bib/abstracts/papers/aaai02b.pdf
# -------------------------------
class DStarLite:
    def __init__(self, start_pos, goal_pos, walls, obs):
        """
        Initializes the search problem by setting up the g-values, rhs-values, and priority queue.
        """
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
        # 04. Set rhs(s) and g(s) to infinity for all s in the state space
        # 05. Set rhs(s_goal) = 0
        # 06. Insert s_goal into U with key [h(s_start, s_goal); 0]
        for x in range(GRID_SIZE - 1):
            for y in range(GRID_SIZE - 1):
                s = Node(x+1, y+1)
                if s.x == self.goal_pos[0] and s.y == self.goal_pos[1]:
                    s.rhs = 0
                    self.U.insert(s, [self.heuristic(self.start, self.goal)])
                    self.goal = s
                if s.x == self.start_pos[0] and s.y == self.start_pos[1]:
                    self.start = s
                self.nodes.append(s)

    def successors(self, s):
        x = s.x
        y = s.y
        return[Node(x-1, y, s), Node(x+1, y, s), Node(x, y-1, s), Node(x, y+1, s),
               Node(x-1, y-1, s), Node(x-1, y+1, s), Node(x+1, y-1, s), Node(x+1, y+1, s)]  
    
    def predecessors(self, s):
        #TODO: implement predecessors
        # if it's parent node predecessors:
        parents = []
        curr = s
        while curr.parent != None:
            parents.append(curr.parent)
            curr = curr.parent
        return parents
        #return self.successors(s)
    
    def compute_rhs(self, s):
        if s == self.start:
            return 0
        return min(self.heuristic(s, s_prime) + self.g[s_prime] for s_prime in self.successors(s))

    def scan_for_changes(self, u):
        """
        Scans the environment for changes in edge costs by comparing the known grid with the actual grid.

        Inputs:
            - u: The current node of the robot.

        Outputs:
            - changed_edges: A dictionary where keys are tuples (u, v) representing edges, and values are the old costs.
        """
        changed_edges = {}

        # Loop through all successors of the current node u
        for v in self.successors(u):
            # Get the old cost of the edge (u, v)
            c_old = self.cost(u, v)

            # Check if the grid cell corresponding to v has changed
            x, y = v.x, v.y
            if self.known_grid[x, y] != self.actual_grid[x, y]:
                # Update the known grid to reflect the current state
                self.known_grid[x, y] = self.actual_grid[x, y]

                # If the grid cell has changed, add the edge (u, v) and its old cost to the dictionary
                changed_edges[(u, v)] = c_old

        return changed_edges

    def heuristic(node, goal):
        '''
        Chebyshev Distance function is admissable and consistent for diagonal grid moves

        Inputs
            node: Node Class
            goal: Node Class
        
        Outputs
            dist: Chebyshev distance metric (units)
        '''
        return max(abs(node.x - goal.x), abs(node.y - goal.y))

    def cost(self, s, s_prime):
        """
        Computes the cost of moving from node s to node s_prime.

        Inputs:
            - s: Current node (x, y)
            - s_prime: Neighboring node (x', y')

        Outputs:
            - cost: Cost of moving from s to s_prime (1 if traversable, infinity if blocked)
        """
        # Check if the edge is blocked by an obstacle
        for wall in self.walls:
            if wall.intersects(LineString([(s.x, s.y), (s_prime.x, s_prime.y)])):
                return float('inf')  # Blocked edge
            
        for obs in self.obstacles:
            if obs.intersects(LineString([(s.x, s.y), (s_prime.x, s_prime.y)])):
                return float('inf')  # Blocked edge
            
        return 1  # Traversable edge

    def calculate_key(self, s):
        """
        Calculates the key for a given vertex s, which is used to prioritize vertices in the priority queue.

        Inputs:
            - s: The vertex for which the key is to be calculated.

        Outputs:
            - key: A tuple (k1, k2) representing the priority of the vertex in the priority queue.

        Explanation:
            - The key is a vector with two components:
                - k1 = min(g(s), rhs(s)) + h(s, s_start) + k_m
                - k2 = min(g(s), rhs(s))
            - The key is used to determine the order in which vertices are expanded in the priority queue.
            - The first component k1 is a heuristic estimate of the total cost from the start to the goal via s.
            - The second component k2 is used to break ties when k1 values are equal.
        """
        k1 = min(s.g, s.rhs) + self.heuristic(self.start, s) + self.k_m
        k2 = min(s.g, s.rhs)
        return [k1, k2]

    def update_vertex(self, u):
        """
        Updates the rhs-value and key of a vertex u and manages its presence in the priority queue.

        Inputs:
            - u: The vertex to be updated.

        Outputs:
            - None (modifies global variables such as rhs, g, U)

        Explanation:
            - If s is not the goal vertex, its rhs-value is updated to the minimum of the sum of g-values of its successors and the edge costs.
            - If the vertex is locally inconsistent (g(s) != rhs(s)), it is inserted into the priority queue with its new key.
            - If the vertex is locally consistent, it is removed from the priority queue.
            - This function ensures that the priority queue only contains vertices that need to be updated.
        """
        # 08. If g(u) ≠ rhs(u) and u is in U, update U with the new key
        # 09. Else if g(u) ≠ rhs(u) and u is not in U, insert u into U
        # 10. Else if g(u) == rhs(u) and u is in U, remove u from U
        if u.g != u.rhs and u in self.U:
            key = self.calculate_key(u)
            self.U.update(u, key)
        
        elif u.g != u.rhs and u not in self.U:
            key = self.calculate_key(u)
            self.U.insert(u, key)
        
        elif u.g == u.rhs and u in self.U:
            self.U.remove(u)

    def compute_shortest_path(self):
        """
        Computes the shortest path from the current start vertex s_start to the goal vertex s_goal.

        Inputs:
            - None (uses global variables such as U, s_start, s_goal, g, rhs, etc.)

        Outputs:
            - None (modifies global variables)

        Explanation:
            - Expands vertices from the priority queue U in order of their keys until the start vertex s_start is locally consistent and its key is less than or equal to the smallest key in U.
            - If a vertex is locally overconsistent (g(s) > rhs(s)), its g-value is set to its rhs-value, making it locally consistent.
            - If a vertex is locally underconsistent (g(s) < rhs(s)), its g-value is set to infinity, making it either locally consistent or overconsistent.
            - The rhs-values and keys of the affected vertices are updated, and they are added to or removed from the priority queue as necessary.
            - This function ensures that the shortest path is always up-to-date with the current state of the graph.
        """
         # 12. While U.TopKey() is smaller than CalculateKey(s_start) OR rhs(s_start) > g(s_start):
        while self.U.top_key() < self.calculate_key(self.start) or self.start.rhs > self.start.g:
            # 13. Extract u = U.Top()
            u = self.U.top()

            # 14. Store old key k_old = U.TopKey()
            k_old = self.U.top_key()

            # 15. Compute new key k_new = CalculateKey(u)
            k_new = self.calculate_key(u)

            # 16. If k_old < k_new, update U with the new key and continue
            # 17. Else if g(u) > rhs(u), update g(u) to rhs(u) and remove u from U
            if k_old < k_new:
                self.U.update(u, k_new)
            elif u.g > u.rhs:
                u.g = u.rhs
                self.U.remove(u)

                # 18. Loop through all predecessors s of u
                for s in self.predecessors(u):
                    # 19. If s is not the goal, update rhs(s) to min(rhs(s), c(s, u) + g(u))
                    if s != self.goal:
                        list = []
                        for sp in self.successors(s):
                            list.append(self.cost(s, sp) + sp.g)
                        s.rhs = min(list)
                
                    # 20. Call UpdateVertex(s) to update U
                    self.update_vertex(s)
                
            else:
                # 21. Else (if g(u) == rhs(u)), store the old g(u) value
                g_old = u.g

                # 22. Set g(u) = infinity
                u.g = float('inf')

                # 23. Loop through all predecessors s of u (including u)
                for s in self.predecessors(u):
                    if s.rhs == self.cost(s, u) + g_old:
                        if s!=self.goal:
                            list = []
                            for sp in self.successors(s):
                                list.append(self.cost(s, sp) + sp.g)
                            s.rhs = min(list)
                
                    # 20. Call UpdateVertex(s) to update U
                    self.update_vertex(s)
                # 24. If rhs(s) was computed from g(u), update rhs(s) to min alternative cost
                # 25. If s is not the goal, update rhs(s) to min cost among successors
                # 26. Call UpdateVertex(s) to update U

    def plan(self):
        """
        The main function that controls the robot's navigation and replanning process.

        Inputs:
            - None (uses global variables such as s_start, s_goal, U, etc.)

        Outputs:
            - None (modifies global variables and controls the robot's movement)

        Explanation:
            - Initializes the search problem by calling Initialize().
            - Computes the initial shortest path by calling ComputeShortestPath().
            - Moves the robot along the computed path.
            - If the robot detects changes in edge costs (e.g., discovers new obstacles), it updates the affected vertices and recalculates the shortest path by calling ComputeShortestPath() again.
            - The process repeats until the robot reaches the goal vertex s_goal.
            - This function orchestrates the entire navigation and replanning process.
        """
        # 28. Set s_last = s_start
        s_last = self.start

        # 30. Call ComputeShortestPath() to find initial shortest path
        self.compute_shortest_path()

        # 31. While s_start is not s_goal:
        while self.start != self.goal:
            # 32. If rhs(s_start) is infinity, there is no known path, return failure
            if self.start.rhs == float('inf'):
                print('No known path... Returning failure')

            # 33. Choose the best next state s_start based on minimum cost to successors
            min_cost = float('inf')
            best_successor = None

            # Loop through all successors of the current start node
            for s_prime in self.successors(self.start):
                # Compute the cost to move to s_prime: c(s, s') + g(s')
                cost = self.cost(self.start, s_prime) + s_prime.g
                # Update the best successor if this cost is smaller
                if cost < min_cost:
                    min_cost = cost
                    best_successor = s_prime

            # Update the start node to the best successor
            if best_successor is not None:
                self.start = best_successor
            else:
                print("No valid successor found. Path planning failed.")
                break
            # 34. Move to s_start (simulate agent movement)
            Robot.move(self.start)

            # 35. Scan for changed edge costs in the environment
            changed_edges = self.scan_for_changes(self.start)

            # 36. If any edge costs have changed:
            if changed_edges:

                # 37. Update km by adding the heuristic distance from s_last to s_start
                self.k_m += self.heuristic(s_last, self.start)

                # 38. Update s_last to s_start
                s_last = self.start

                # 39. Loop through all directed edges (u, v) that changed:
                for (u, v), c_old in changed_edges.items():
                    # 41. Update c(u, v) with the new cost
                    c_new = self.update_cost(u, v)
                    
                    # 42. If cold > c(u, v), update rhs(u) with new minimal cost path
                    if c_old > c_new:
                        if u != self.goal:
                            u.rhs = min(u.rhs, self.cost(u, v) + v.g)
                        
                    # 43. Else if rhs(u) was derived from old cost, update rhs(u)
                    elif u.rhs == c_old + v.g:
                            if u != self.goal:
                                list = []
                                for sp in self.successors(u):
                                    list.append(self.cost(u, sp) + sp.g)
                                u.rhs = min(list)

                    # 45. Call UpdateVertex(u) to update U with new cost
                    self.update_vertex(u)

                # 46. Call ComputeShortestPath() again to adjust path planning
                self.compute_shortest_path

# -------------------------------
# Robot Class
# -------------------------------
class Robot:
    def __init__(self, pos):
        self.x, self.y = pos

    def move(self, newpoint):
        self.x, self.y = newpoint

# -------------------------------
# Main Class
# -------------------------------
class Main:
    def __init__(self, map_name="maze"):
        self.walls = WALLS
        self.obs = OBS_CONFIGS.get(map_name, OBS_CONFIGS["maze"])

    def update_plot(self):
        # Initialize plot
        x = np.linspace(0, GRID_SIZE, GRID_SIZE)
        y = np.linspace(0, GRID_SIZE, GRID_SIZE)
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots()
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_aspect('equal', 'box')

        # Plot Grid
        ax.clear()
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_aspect('equal', 'box')
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, color='blue', lw=0.5))

        # Known map
        for wall in self.walls:
            x, y = wall.exterior.xy
            ax.fill(x, y, alpha=1, fc='black')

        # Unkown map of obstacles
        for obs in self.obs:
            x, y = obs.exterior.xy
            ax.fill(x, y, alpha=.5, fc='black')

        plt.draw()
        plt.pause(0.1)
        plt.show()

if __name__ == "__main__":
    main = Main()
    main.update_plot()