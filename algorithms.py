import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import numpy as np
from utils import is_point_inside_obstacle, distance, nearest_neighbors, \
    nearest_neighbor, is_collision_free, Node
from scipy.spatial import distance_matrix
from debug_utils import plot_samples_and_obstacles

def fmt_star(start, goal, obstacles, Nmax, eta):
    """
    Compute a path from a start point to a goal region, avoiding obstacles, using the FMT* algorithm.

    Parameters:
    - start (tuple): A tuple (x, y) representing the starting point of the path.
    - goal (list): A list of tuples [(x1, y1), (x2, y2), ...], representing vertices of the polygon defining the goal region.
    - obstacles (list): A list of obstacles, each represented by a tuple (x, y, width, height).
    - Nmax (int): The maximum number of random samples to generate in the configuration space.
    - eta (float): A scaling factor used to determine the connection radius in the FMT* algorithm.

    Returns:
    - path (list): A list of tuples [(x1, y1), (x2, y2), ...], representing the found path from start to goal.
    - V (list): A list of Node objects, representing all sampled nodes in the configuration space.

    Notes:
    - The function assumes a 2D configuration space and works with Euclidean distances.
    - The goal is defined as a polygon and the path is considered found when any node within this polygon is reached.
    - Obstacles are assumed to be axis-aligned rectangles.
    - The function utilizes several helper functions (e.g., `is_point_inside_obstacle`, `nearest_neighbors`, etc.) which need to be defined elsewhere in the code.
    - The function may print progress information (iterations, nodes in tree, unvisited nodes) to the console during execution.
    """

    start_node = Node(start)
    goal_node = Node(goal)
    start_node.cost = 0
    
    # Filter samples to remove those inside obstacles
    rand_samples = [(np.random.uniform(0, 1), np.random.uniform(0, 1)) for _ in range(Nmax)]
    collision_free_samples = [sample for sample in rand_samples if not is_point_inside_obstacle(sample, obstacles)]
    
    V = [start_node] + [Node(sample) for sample in collision_free_samples]
    N = len(V)
    
    # Compute mu and r
    mu = 1  # Assuming a 1x1 unit square
    for obstacle in obstacles:
        mu -= obstacle[2] * obstacle[3]  # Subtracting the area of each obstacle
    r = eta * np.sqrt(2 * mu / np.pi * np.log(N) / N) 
    
    # Precompute nearest neighbors
    NN = [[V.index(nn) for nn in nearest_neighbors(V, v.point, r)] for v in V]

    # Assuming goal is a list of points defining the polygon [(x1, y1), (x2, y2), ...]
    goal_polygon = Polygon(goal)
    
    # Initialize variables
    W = np.ones(N, dtype=bool)
    H = np.zeros(N, dtype=bool)
    C = np.full(N, np.inf)
    W[0] = False
    H[0] = True
    C[0] = 0
    z = 0
    
    iters = 0
    edges = []  # To store the edges in the order they are created
    # Main loop
    while not goal_polygon.contains(Point(V[z].point)):
        H_new = []
        # For each nearest neighbor of point z that is unprocessed (i.e., in W)
        for x_idx in [i for i in NN[z] if W[i]]:
            Y_near = [i for i in NN[x_idx] if H[i]] # All points in H (visited) that are nearest neighbors of x_idx (i.e., all nearest neighbors of z)
            if Y_near:  # Check if Y_near is not empty
                y_min_idx = Y_near[np.argmin([C[y] + distance(V[x_idx].point, V[y].point) for y in Y_near])]
                if is_collision_free(V[x_idx].point, V[y_min_idx].point, obstacles):
                    V[x_idx].parent = V[y_min_idx]
                    edges.append((V[y_min_idx], V[x_idx]))  # Store the edge
                    C[x_idx] = C[y_min_idx] + distance(V[x_idx].point, V[y_min_idx].point)
                    H_new.append(x_idx)
                    W[x_idx] = False    # Mark x_idx as processed
        H[H_new] = True
        H[z] = False    # z has been visited
        z = np.where(H)[0][np.argmin(C[H])]
    
        iters += 1
        print(f"== FMT★ ==Iteration: {iters}, Nodes in Tree: {np.sum(H)}, Unvisited Nodes: {np.sum(W)}")
    
    # Reconstruct path
    path = []
    total_cost = 0  # Initialize total_cost
    while V[z].parent is not None:
        path.append(V[z].point)
        # Add the cost from the current node to its parent to total_cost
        total_cost += distance(V[z].point, V[z].parent.point)
        z = V.index(V[z].parent)
    path.append(start)
    path.reverse()

    return round(total_cost, 3), path, edges, 'fmtstar'

def rrt_star(start, goal, obstacles, Nmax, eta):
    """
    Compute a path from a start point to a goal region, avoiding obstacles, using the RRT* algorithm.

    Parameters:
    - start (tuple): A tuple (x, y) representing the starting point of the path.
    - goal (list): A list of tuples [(x1, y1), (x2, y2), ...], representing vertices of the polygon defining the goal region.
    - obstacles (list): A list of obstacles, each represented by a tuple (x, y, width, height).
    - Nmax (int): The maximum number of nodes to be added to the tree.
    - eta (float): A parameter to determine the maximum step size during the extension of the tree.

    Returns:
    - path (list): A list of tuples [(x1, y1), (x2, y2), ...], representing the found path from start to goal.
    - V (list): A list of Node objects, representing all nodes in the tree.

    Notes:
    - The function assumes a 2D configuration space and works with Euclidean distances.
    - The goal is defined as a polygon and the path is considered found when any node within this polygon is reached.
    - Obstacles are assumed to be axis-aligned rectangles.
    - The RRT* algorithm is an asymptotically optimal variant of the Rapidly-exploring Random Tree (RRT) algorithm, which incrementally 
    builds a space-filling tree while keeping track of the cost-to-come and ensuring that the solution path converges towards the optimal 
    solution with an increasing number of samples.
    """
    start_node = Node(start)
    start_node.cost = 0
    
    V = [start_node]
    A = [None] * Nmax
    C = [float('inf')] * Nmax
    C[0] = 0
    
    mu = 1
    for obstacle in obstacles:
        # Assuming obstacle format: (x, y, width, height)
        mu -= obstacle[2] * obstacle[3]
    
    r = lambda N: min(1.1*np.sqrt(3*mu/np.pi*np.log(N)/N), eta)
    
    for k in range(Nmax):
        q = tuple(np.random.rand(2))
        q_nrst, nrst_idx = nearest_neighbor(V, q)
        q_new = tuple(np.array(q_nrst.point) + min(eta/np.linalg.norm(np.array(q)-np.array(q_nrst.point)), 1)*(np.array(q)-np.array(q_nrst.point)))
        
        if is_collision_free(q_new, q_nrst.point, obstacles):
            near_idx = nearest_neighbors(V, q_new, r(len(V)))
            N = len(V)
            V.append(Node(q_new))
            min_idx = nrst_idx
            min_c = C[nrst_idx] + np.linalg.norm(np.array(q_new)-np.array(q_nrst.point))
            
            for near_node in near_idx:
                if C[V.index(near_node)] + np.linalg.norm(np.array(q_new)-np.array(near_node.point)) < min_c and \
                   is_collision_free(q_new, near_node.point, obstacles):
                    min_idx = V.index(near_node)
                    min_c = C[V.index(near_node)] + np.linalg.norm(np.array(q_new)-np.array(near_node.point))
            
            A[N] = min_idx
            C[N] = min_c
            V[N].parent = V[min_idx]  # Set parent
            
            for near_node in near_idx:
                if C[N] + np.linalg.norm(np.array(q_new)-np.array(near_node.point)) < C[V.index(near_node)] and \
                   is_collision_free(q_new, near_node.point, obstacles):
                    A[V.index(near_node)] = N
                    C[V.index(near_node)] = C[N] + np.linalg.norm(np.array(q_new)-np.array(near_node.point))
    
    # Construct path
    goal_polygon = Polygon(goal)
    goal_pts = [i for i, v in enumerate(V) if goal_polygon.contains(Point(v.point))]
    goal_costs = [C[i] for i in goal_pts]
    min_cost_idx = goal_pts[np.argmin(goal_costs)]

    path = []
    total_cost = 0  # Initialize total_cost
    current_idx = min_cost_idx  # Start from the goal node
    while A[current_idx] is not None:
        path.append(V[current_idx].point)
        # Add the cost from the current node to its parent to total_cost
        total_cost += np.linalg.norm(np.array(V[current_idx].point) - np.array(V[A[current_idx]].point))
        current_idx = A[current_idx]
    path.append(start)
    path.reverse()

    return round(total_cost, 3), path, V, 'rrtstar'

def rrt(start, goal, obstacles, Nmax, eta):
    """
    Compute a path from a start point to a goal region, avoiding obstacles, using the RRT algorithm.

    Parameters:
    - start (tuple): A tuple (x, y) representing the starting point of the path.
    - goal (list): A list of tuples [(x1, y1), (x2, y2), ...], representing vertices of the polygon defining the goal region.
    - obstacles (list): A list of obstacles, each represented by a tuple (x, y, width, height).
    - Nmax (int): The maximum number of nodes to be added to the tree.
    - eta (float): A parameter to determine the maximum step size during the extension of the tree.

    Returns:
    - path (list): A list of tuples [(x1, y1), (x2, y2), ...], representing the found path from start to goal.
    - V (list): A list of Node objects, representing all nodes in the tree.
    """
    start_node = Node(start)
    V = [start_node]  # List of nodes
    goal_polygon = Polygon(goal)
    
    for k in range(Nmax):
        q = np.random.rand(2)
        q_near, nn_idx = nearest_neighbor(V, q)
        q_new_coord = tuple(q_near.point + (q - np.array(q_near.point)) * min(eta/np.linalg.norm(q - np.array(q_near.point)), 1))
        
        if is_collision_free(q_new_coord, q_near.point, obstacles):
            q_new = Node(q_new_coord)
            q_new.parent = q_near
            V.append(q_new)
            
            if goal_polygon.contains(Point(q_new.point)):
                break
    
    # Reconstruct path
    path = []
    cost = 0
    while V[-1].parent is not None:
        path.append(V[-1].point)
        if V[-1].parent is not None:
            cost += np.linalg.norm(np.array(V[-1].point) - np.array(V[-1].parent.point))
        V[-1] = V[-1].parent
    path.append(start)
    path.reverse()
    
    return round(cost, 3), path, V, 'rrt'

def prm_star(start, goal, obstacles, Nmax, eta):
    start_node = Node(start)
    start_node.cost = 0
    
    rand_samples = [(np.random.uniform(0, 1), np.random.uniform(0, 1)) for _ in range(Nmax)]
    collision_free_samples = [sample for sample in rand_samples if not is_point_inside_obstacle(sample, obstacles)]
    
    V = [start_node] + [Node(sample) for sample in collision_free_samples]
    N = len(V)
    
    mu = 1  # Assuming a 1x1 unit square
    for obstacle in obstacles:
        mu -= obstacle[2] * obstacle[3]
    
    r = eta * np.sqrt(2 * mu / np.pi * np.log(N) / N)
    
    points = np.array([v.point for v in V])
    D = distance_matrix(points, points)
    
    edges = []  # Store edges as they are created

    for i, v in enumerate(V):
        v_neighbors = [u for j, u in enumerate(V) if D[i, j] < r and i != j and is_collision_free(v.point, u.point, obstacles)]
        for u in v_neighbors:
            if v.cost + distance(v.point, u.point) < u.cost:
                u.parent = v
                u.cost = v.cost + distance(v.point, u.point)
                edges.append((v, u))  # Store the edge
    
    start_neighbors = [u for j, u in enumerate(V) if D[0, j] < r and is_collision_free(start, u.point, obstacles)]
    for u in start_neighbors:
        if start_node.cost + distance(start, u.point) < u.cost:
            u.parent = start_node
            u.cost = start_node.cost + distance(start, u.point)
            edges.append((start_node, u))  # Store the edge
    
    goal_polygon = Polygon(goal)
    goal_nodes = [v for v in V if goal_polygon.contains(Point(v.point))]
    
    if not goal_nodes:
        return [], edges  # No path found
    
    goal_node = min(goal_nodes, key=lambda v: v.cost)
    
    path = []
    current = goal_node
    while current.parent is not None:
        path.append(current.point)
        current = current.parent
    path.append(start)
    path.reverse()
    
    return round(goal_node.cost, 3), path, edges, 'prmstar'