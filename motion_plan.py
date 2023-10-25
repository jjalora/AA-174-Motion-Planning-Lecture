import numpy as np
import matplotlib.pyplot as plt
from algorithms import fmt_star, rrt_star, rrt, prm_star
from utils import plot_results, animate_edges_and_path, generateValidationSamples, \
    animate_rrt_star, visualize_prm_star

# Example 1: Two Boxes
example = 1
start = (0, 0)
# Define obstacles as polygons
obstacles = [
    (0.1, 0.1, 0.1, 0.1),
    (0.6, 0.6, 0.2, 0.2)
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
k = 500 # 50
seed = 962 #899, 904, 962

# Generate validation samples to check with MATLAB
pathToMatlab = "/Users/jalora/Desktop/AA-203-MP-Lecture"
generateValidationSamples(obstacles, k, seed, path=pathToMatlab)

# FMTstar Implementation
eta = 1.5
cost, path, edges, V_near, label = fmt_star(start, goal, obstacles, k, eta, seed=seed)

# RRTstar and RRT Implementation
# eta = 0.1
# cost, path, V_near, label = rrt(start, goal, obstacles, k, eta, seed=seed)

# eta = 0.3
# cost, path, V, A, added_edges, rewired_edges, label = rrt_star(start, goal, obstacles, k, eta, seed=seed)

# PRM star Implementation
# For Example 1
# eta = 1.5
# For example 2
# eta = 3.0

# cost, path, edges, V, label = prm_star(start, goal, obstacles, k, eta, seed=seed)


# Animate and plot results
params = {'k': k, 'eta': eta}
if label == "fmtstar":
    animate_edges_and_path(start, obstacles, goal, path, label, cost, edges=edges, V_near=V_near, params=params, example=example, static_nodes=True)
elif label =="rrtstar":
    params = {'V': V, 'A': A, 'added_edges': added_edges, 'rewired_edges': rewired_edges}
    animate_rrt_star(start, goal, obstacles, path, cost, params)
elif label == "prmstar":
    visualize_prm_star(start, goal, obstacles, path, edges, cost, V)
else:
    animate_edges_and_path(start, obstacles, goal, path, label, cost, V_near=V_near, params=params, example=example)