import numpy as np
import matplotlib.pyplot as plt
from algorithms import fmt_star, rrt_star, rrt, prm_star
from utils import plot_results, animate_edges_and_path

# Example 1: Two Boxes
example = 1
start = (0, 0)
# Define obstacles as polygons
obstacles = [
    (0.1, 0.1, 0.3, 0.3),
    (0.4, 0.6, 0.5, 0.3)
]
# Define goal as a polygon
goal = [(0.9, 0.9), (1, 0.9), (1, 1), (0.9, 1)]

# Example 2: Windy Maze
# example = 2
# start = (0.1, 0.65)

# # Define obstacles as polygons
# obstacles = [
#     (0, 0.45, 0.9, 0.1),
#     (0.25, 0.2, 0.05, 0.6),
#     (0.75, 0.2, 0.05, 0.6),
#     (0.5, 0.7, 0.05, 0.3), 
#     (0.5, 0, 0.05, 0.3)  
# ]

# # Define goal as a polygon
# goal = [
#     (0, 0.3),
#     (0.15, 0.3),
#     (0.15, 0.45),
#     (0, 0.45)
# ]

# Initialize
edges, V_near = None, None
k = 1000
# k = 1800 # for example 2

# FMTstar Implementation
# eta = 1.5
# cost, path, edges, label = fmt_star(start, goal, obstacles, k, eta)

# RRTstar and RRT Implementation
# eta = 0.1
# cost, path, V_near, label = rrt(start, goal, obstacles, k, eta)
# cost, path, V_near, label = rrt_star(start, goal, obstacles, k, eta)

# PRM star Implementation
# For Example 1
eta = 2.5
# For example 2
# eta = 3.0

cost, path, edges, label = prm_star(start, goal, obstacles, k, eta)


# Animate and plot results
params = {'k': k, 'eta': eta}
if label == "fmtstar":
    animate_edges_and_path(start, obstacles, goal, path, label, cost, edges=edges, params=params, example=example)
elif label == "prmstar":
    plot_results(start, obstacles, goal, path, label, cost, edges=edges, params=params, example=example)
else:
    animate_edges_and_path(start, obstacles, goal, path, label, cost, V_near=V_near, params=params, example=example)