import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

# Define grid size and create grid
grid_size = 25
x = np.linspace(0, grid_size, grid_size)
y = np.linspace(0, grid_size, grid_size)
X, Y = np.meshgrid(x, y)

# Initialize plot
fig, ax = plt.subplots()
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_aspect('equal', 'box')

# Function to plot grid
def plot_grid():
    ax.clear()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal', 'box')
    
    for i in range(grid_size):
        for j in range(grid_size):
            ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, color='blue', lw=0.5))

# Function to create and plot a polygon using shapely
def plot_polygon():
    # Example polygon with shapely
    polygon = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
    
    x, y = polygon.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='orange', edgecolor='r')

# Initial grid plot
plot_grid()
plot_polygon()

# Show plot
plt.draw()
plt.pause(0.1)

# Update grid
for _ in range(100):  # For example, run 100 updates
    plot_grid()
    plot_polygon()  # For testing, update with a new polygon each time
    plt.draw()
    plt.pause(0.1)

plt.show()
