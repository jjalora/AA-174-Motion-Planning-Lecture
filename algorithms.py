import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import numpy as np
from utils import is_point_inside_obstacle, distance, nearest_neighbors, \
    nearest_neighbor, is_collision_free, Node
from scipy.spatial import distance_matrix
from debug_utils import plot_samples_and_obstacles
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import heapq

def fmt_star(start, goal, obstacles, Nmax, eta, seed=None):
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

    # Seed the random number generator
    if seed is not None:
        np.random.seed(seed)
    
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
        # print(f"== FMT★ ==Iteration: {iters}, Nodes in Tree: {np.sum(H)}, Unvisited Nodes: {np.sum(W)}")
    
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

    return round(total_cost, 3), path, edges, V, 'fmtstar'

def rrt_star(start, goal, obstacles, Nmax, eta, seed=None):
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

    # Seed the random number generator
    if seed is not None:
        np.random.seed(seed)

    V = [start_node]
    A = [None] * Nmax
    C = [float('inf')] * Nmax
    C[0] = 0
    R = [(np.random.uniform(0, 1), np.random.uniform(0, 1)) for _ in range(Nmax)]

    rewired_edges = []
    added_edges = []

    mu = 1
    for obstacle in obstacles:
        mu -= obstacle[2] * obstacle[3]

    r = lambda N: min(1.1*np.sqrt(3*mu/np.pi*np.log(N)/N), eta)

    for k in range(Nmax):
        q = R[k]
        q_nrst, nrst_idx = nearest_neighbor(V, q)
        q_new = tuple(np.array(q_nrst.point) + min(eta/np.linalg.norm(np.array(q)-np.array(q_nrst.point)), 1)*(np.array(q)-np.array(q_nrst.point)))

        if is_collision_free(q_new, q_nrst.point, obstacles):
            near_nodes = nearest_neighbors(V, q_new, r(len(V)))
            N = len(V)
            V.append(Node(q_new))
            min_idx = nrst_idx
            min_c = C[nrst_idx] + np.linalg.norm(np.array(q_new) - np.array(q_nrst.point))
            
            near_norms = [np.linalg.norm(np.array(node.point) - np.array(q_new)) for node in near_nodes]
            sorted_costs = sorted([(cost + C[V.index(node)], i) for i, (node, cost) in enumerate(zip(near_nodes, near_norms))], key=lambda x: x[0])

            for cost, i in sorted_costs:
                if cost >= min_c:
                    break
                if is_collision_free(q_new, near_nodes[i].point, obstacles):
                    min_idx = V.index(near_nodes[i])
                    min_c = cost
                    break

            A[N] = min_idx
            C[N] = min_c
            V[N].parent = V[min_idx]
            added_edges.append((V[min_idx].point, V[N].point))

            near_norms = [np.linalg.norm(np.array(node.point) - np.array(q_new)) for node in near_nodes]
            rewire_nodes = [near_nodes[i] for i, dist in enumerate(near_norms) if C[N] + dist < C[V.index(near_nodes[i])]]

            for near_node in rewire_nodes:
                dist = np.linalg.norm(np.array(q_new) - np.array(near_node.point))
                if is_collision_free(q_new, near_node.point, obstacles):
                    if near_node.parent:  
                        rewired_edges.append((tuple(near_node.parent.point), tuple(near_node.point)))
                    A[V.index(near_node)] = N
                    c_delta = C[N] + dist - C[V.index(near_node)]
                    C[V.index(near_node)] += c_delta
                    added_edges.append((tuple(V[N].point), tuple(near_node.point)))

                    descendants_to_process = [V.index(near_node)]
                    while descendants_to_process:
                        current_node_idx = descendants_to_process.pop()
                        children_indices = [V.index(child) for child in V if A[V.index(child)] == current_node_idx]
                        descendants_to_process.extend(children_indices)

                        for child_idx in children_indices:
                            C[child_idx] += c_delta

    # Construct path
    goal_polygon = Polygon(goal)
    goal_pts = [i for i, v in enumerate(V) if goal_polygon.contains(Point(v.point))]

    if not goal_pts:
        return float('inf'), [], V, A, added_edges, rewired_edges, 'rrtstar'

    goal_costs = [C[i] for i in goal_pts]
    min_cost_idx = goal_pts[np.argmin(goal_costs)]

    path = []
    total_cost = 0
    current_idx = min_cost_idx
    while A[current_idx] is not None:
        path.append(V[current_idx].point)
        total_cost += np.linalg.norm(np.array(V[current_idx].point) - np.array(V[A[current_idx]].point))
        current_idx = A[current_idx]
    path.append(start)
    path.reverse()

    return round(total_cost, 3), path, V, A, added_edges, rewired_edges, 'rrtstar'

def rrt(start, goal, obstacles, Nmax, eta, seed=None):
    """
    Compute a path from a start point to a goal region, avoiding obstacles, using the RRT algorithm.

    Parameters:
    - start (tuple): A tuple (x, y) representing the starting point of the path.
    - goal (list): A list of tuples [(x1, y1), (x2, y2), ...], representing vertices of the polygon defining the goal region.
    - obstacles (list): A list of obstacles, each represented by a tuple (x, y, width, height).
    - Nmax (int): The maximum number of nodes to be added to the tree.
    - eta (float): A parameter to determine the maximum step size during the extension of the tree.

    Returns:
    - cost (float): The cost of the found path.
    - path (list): A list of tuples [(x1, y1), (x2, y2), ...], representing the found path from start to goal.
    - V (list): A list of Node objects, representing all nodes in the tree.
    """
    start_node = Node(start)
    V = [start_node]  # List of nodes
    goal_polygon = Polygon(goal)

    # Seed the random number generator
    if seed is not None:
        np.random.seed(seed)
    
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
    
    # Check if path reaches the goal
    if not goal_polygon.contains(Point(V[-1].point)):
        return float('inf'), [], V, 'rrt'
    
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

def prm_star(start, goal, obstacles, Nmax, eta, seed=None):
    if seed is not None:
        np.random.seed(seed)

    start_node = Node(start)
    start_node.cost = 0
    V = [start_node]  # Initialize with the start node
    
    # Sample and add to V
    for _ in range(Nmax):
        rand_point = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        if not is_point_inside_obstacle(rand_point, obstacles):
            V.append(Node(rand_point))
    
    N = len(V)
    mu = 1
    for obstacle in obstacles:
        mu -= obstacle[2] * obstacle[3]

    r = eta * np.sqrt(2 * mu / np.pi * np.log(N) / N)
    points = np.array([v.point for v in V])
    D = distance_matrix(points, points)
    
    # Create adjacency list for the graph
    adj_list = {v: [] for v in V}
    
    for i, v in enumerate(V):
        v_neighbors = [V[j] for j in range(i+1, N) if D[i, j] < r and is_collision_free(v.point, V[j].point, obstacles)]
        for u in v_neighbors:
            adj_list[v].append((u, D[i, V.index(u)]))
            adj_list[u].append((v, D[V.index(u), i]))  # Because the graph is undirected
    
    # Dijkstra's algorithm
    visited = set()
    pq = [(0, start_node)]
    while pq:
        current_cost, current_node = heapq.heappop(pq)
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        for neighbor, edge_cost in adj_list[current_node]:
            if neighbor not in visited:
                new_cost = current_cost + edge_cost
                if new_cost < neighbor.cost:
                    neighbor.cost = new_cost
                    neighbor.parent = current_node
                    heapq.heappush(pq, (new_cost, neighbor))
    
    goal_nodes = [v for v in V if Polygon(goal).contains(Point(v.point))]
    if not goal_nodes:
        return float('inf'), [], [(k, v[0]) for k, values in adj_list.items() for v in values], V, 'prmstar'
    
    goal_node = min(goal_nodes, key=lambda v: v.cost)
    path = []
    current = goal_node
    while current:
        path.append(current.point)
        current = current.parent
    path.reverse()
    
    return round(goal_node.cost, 3), path, [(k, v[0]) for k, values in adj_list.items() for v in values], V, 'prmstar'