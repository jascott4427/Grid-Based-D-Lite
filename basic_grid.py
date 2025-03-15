import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

# Define grid size and create grid
GRID_SIZE = 25

# Define different wall configurations
WALL_CONFIGS = {
    "simple": [
        Polygon([(0, 0), (GRID_SIZE, 0), (GRID_SIZE, 1), (0, 1)]),
        Polygon([(0, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(0, 0), (1, 0), (1, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(GRID_SIZE - 1, 0), (GRID_SIZE, 0), (GRID_SIZE, GRID_SIZE), (GRID_SIZE - 1, GRID_SIZE)])
    ],
    "obstacles": [
        Polygon([(0, 0), (GRID_SIZE, 0), (GRID_SIZE, 1), (0, 1)]),
        Polygon([(0, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(0, 0), (1, 0), (1, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(GRID_SIZE - 1, 0), (GRID_SIZE, 0), (GRID_SIZE, GRID_SIZE), (GRID_SIZE - 1, GRID_SIZE)]),
        Polygon([(10, 10), (15, 10), (15, 15), (10, 15)]),  # Obstacle in the middle
        Polygon([(5, 20), (10, 20), (10, 25), (5, 25)])     # Another obstacle
    ],
    "maze": [
        Polygon([(0, 0), (GRID_SIZE, 0), (GRID_SIZE, 1), (0, 1)]),
        Polygon([(0, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(0, 0), (1, 0), (1, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(GRID_SIZE - 1, 0), (GRID_SIZE, 0), (GRID_SIZE, GRID_SIZE), (GRID_SIZE - 1, GRID_SIZE)]),
        Polygon([(5, 5), (10, 5), (10, 10), (5, 10)]),  # Maze-like walls
        Polygon([(15, 5), (20, 5), (20, 10), (15, 10)]),
        Polygon([(5, 15), (10, 15), (10, 20), (5, 20)]),
        Polygon([(15, 15), (20, 15), (20, 20), (15, 20)])
    ]
}

# -------------------------------
# DStarLite Class based on https://idm-lab.org/bib/abstracts/papers/aaai02b.pdf
# -------------------------------
class DStarLite:
    def __init__(self, goal):
        # Initialize any variables needed to run planner
        self.goal = goal

    def get_successors(self, s):
        x, y = s
        return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]    
    
    def compute_rhs(self, s):
        if s == self.start:
            return 0
        return min(self.cost(s, s_prime) + self.g[s_prime] for s_prime in get_successors(s))

    #TODO: Define rhs(s) & g(s) -> what are these?
    def calculate_key():
        # 02. Initialize U (priority queue) as an empty set
        U = set()   
        # 03. Set km = 0
        k_m = 0
        # 04. Set rhs(s) and g(s) to infinity for all s in the state space
        rhs_s = np.inf
        g_s = np.inf 
        # 05. Set rhs(s_goal) = 0
        rhs_s_goal = 0
        # 06. Insert s_goal into U with key [h(s_start, s_goal); 0]

    def update_vertex(u):
        # 08. If g(u) ≠ rhs(u) and u is in U, update U with the new key
        # 09. Else if g(u) ≠ rhs(u) and u is not in U, insert u into U
        # 10. Else if g(u) == rhs(u) and u is in U, remove u from U

    def compute_shortest_path():
         # 12. While U.TopKey() is smaller than CalculateKey(s_start) OR rhs(s_start) > g(s_start):
        #while (U.TopKey() < calculate_key(s_start) or rhs(s_start) > g(s_start)):
            # 13. Extract u = U.Top()
            #u = U.Top()

            # 14. Store old key k_old = U.TopKey()
            #k_old = U.TopKey()
            # 15. Compute new key k_new = CalculateKey(u)
            # 16. If k_old < k_new, update U with the new key and continue
            # 17. Else if g(u) > rhs(u), update g(u) to rhs(u) and remove u from U
            # 18. Loop through all predecessors s of u
                # 19. If s is not the goal, update rhs(s) to min(rhs(s), c(s, u) + g(u))
                    # 20. Call UpdateVertex(s) to update U
            # 21. Else (if g(u) == rhs(u)), store the old g(u) value
                # 22. Set g(u) = infinity
        # 23. Loop through all predecessors s of u (including u)
            # 24. If rhs(s) was computed from g(u), update rhs(s) to min alternative cost
            # 25. If s is not the goal, update rhs(s) to min cost among successors
            # 26. Call UpdateVertex(s) to update U

    def plan():
        # 28. Set s_last = s_start
        # 29. Call Initialize() function to set up the problem
        # 30. Call ComputeShortestPath() to find initial shortest path
        # 31. While s_start is not s_goal:
            # 32. If rhs(s_start) is infinity, there is no known path, return failure
            # 33. Choose the best next state s_start based on minimum cost to successors
            # 34. Move to s_start (simulate agent movement)
            # 35. Scan for changed edge costs in the environment
            # 36. If any edge costs have changed:
            # 37. Update km by adding the heuristic distance from s_last to s_start
            # 38. Update s_last to s_start
            # 39. Loop through all directed edges (u, v) that changed:
                # 40. Store old edge cost cold = c(u, v)
                # 41. Update c(u, v) with the new cost
                # 42. If cold > c(u, v), update rhs(u) with new minimal cost path
                # 43. Else if rhs(u) was derived from old cost, update rhs(u)
                # 44. If u is not the goal, recompute rhs(u) from its successors
                # 45. Call UpdateVertex(u) to update U with new cost
                # 46. Call ComputeShortestPath() again to adjust path planning
                # 47. Repeat until the goal is reached
                # 48. End of algorithm execution

# -------------------------------
# Main Class
# -------------------------------
class Main:
    def __init__(self, map_name="maze"):
        self.walls = WALL_CONFIGS.get(map_name, WALL_CONFIGS["maze"])
    
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

        # Plot walls as polygons using shapely
        for wall in self.walls:
            x, y = wall.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='orange', edgecolor='r')

        plt.draw()
        plt.pause(0.1)
        plt.show()

if __name__ == "__main__":
    main = Main()
    main.update_plot