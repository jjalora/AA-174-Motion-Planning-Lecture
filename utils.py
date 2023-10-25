import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point
import numpy as np
from debug_utils import plot_samples_and_obstacles
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import csv
from matplotlib.animation import FuncAnimation
from math import sqrt, cos, sin, atan2

from os.path import join, dirname, abspath
PATH = dirname(abspath(__file__))

class Node:
    def __init__(self, point):
        self.point = point
        self.parent = None
        self.cost = float('inf')

def is_point_inside_obstacle(point, obstacles):
    for obstacle in obstacles:
        obstacle_polygon = Polygon([(obstacle[0], obstacle[1]), 
                                    (obstacle[0] + obstacle[2], obstacle[1]), 
                                    (obstacle[0] + obstacle[2], obstacle[1] + obstacle[3]), 
                                    (obstacle[0], obstacle[1] + obstacle[3])])
        if obstacle_polygon.contains(Point(point)):
            return True
    return False

def distance(a, b):
    return np.linalg.norm(np.array(a)-np.array(b))

def nearest_neighbors(nodes, point, r):
    return [node for node in nodes if distance(node.point, point) < r]

def nearest_neighbor(nodes, point):
    min_dist = float('inf')
    nearest = None
    idx = -1
    for i, node in enumerate(nodes):
        dist = distance(node.point, point)
        if dist < min_dist:
            min_dist = dist
            nearest = node
            idx = i
    return nearest, idx

def step_from_to(p1, p2, EPSILON):
    if distance(p1, p2) < EPSILON:
        return p2
    else:
        theta = atan2(p2[1] - p1[1], p2[0] - p1[0])
        return p1[0] + EPSILON * cos(theta), p1[1] + EPSILON * sin(theta)

def chooseParent(nn, newnode, nodes, RADIUS):
    """
    This function sets the parent for the newnode.
    It checks all existing nodes and sets the parent to be the closest node that results in the lowest cost path.
    """
    for p in nodes:
        if distance(p.point, newnode.point) < RADIUS and p.cost + distance(p.point, newnode.point) < nn.cost + distance(nn.point, newnode.point):
            nn = p
    newnode.parent = nn
    newnode.cost = nn.cost + distance(nn.point, newnode.point)
    return newnode, nn


def rewire(nodes, newnode, RADIUS, A):
    """
    This function checks for nearby nodes and tries to rearrange connections to minimize path costs.
    """
    rewired = []
    for node in nodes:
        if node != newnode.parent and distance(node.point, newnode.point) < RADIUS:
            potential_new_cost = newnode.cost + distance(node.point, newnode.point)
            if potential_new_cost < node.cost:
                # Store the old edge before modifying
                old_edge = (node.parent.point, node.point) if node.parent else None

                node.parent = newnode
                node.cost = potential_new_cost
                new_edge = (newnode.point, node.point)

                if old_edge:
                    rewired.append(old_edge)  # Add the old edge to rewired_edges
                    # Remove the old edge from A
                    for edge in A:
                        if edge[1] == node:  # This checks if the node was the child in the tuple
                            A.remove(edge)
                            break
                # Add the new edge to A and rewired_edges
                A.append((newnode, node))
                rewired.append(new_edge)

    return nodes, rewired


def NN1(nodes, q):
    nodes_points = [n.point for n in nodes]
    V = np.array(nodes_points).T
    distances = np.sum((V - np.outer(q, np.ones(V.shape[1])))**2, axis=0)
    nn_idx = np.argmin(distances)
    nn_node = nodes[nn_idx]
    return nn_node, nn_idx

def line_intersects_rect(a, b, rect):
    line = LineString([a, b])
    polygon = Polygon([(rect[0], rect[1]), 
                       (rect[0] + rect[2], rect[1]), 
                       (rect[0] + rect[2], rect[1] + rect[3]), 
                       (rect[0], rect[1] + rect[3])])
    return line.intersects(polygon)

def is_collision_free(a, b, obstacles):
    for obstacle in obstacles:
        if line_intersects_rect(a, b, obstacle):
            return False
    return True

def rectangle_to_polygon_coords(rect):
    x, y, w, h = rect
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

def plot_results(start, obstacles, goal, path, alg_label, cost, edges=None, V_near=None, params=None, example=1):
    alg_titles = {'rrtstar': "RRT٭",
                  'rrt': "RRT",
                  'fmtstar': "FMT٭",
                  'prmstar': "PRM٭"}
    
    plt.figure(figsize=(10, 10))
    plt.axis([0, 1, 0, 1])

    # Plot start and goal
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    goal_polygon = plt.Polygon(goal, edgecolor='g', facecolor=(0, 1, 0, 0.5), label='Goal')
    plt.gca().add_patch(goal_polygon)

    # Plot obstacles
    for obstacle in obstacles:
        obstacle_rect = plt.Rectangle((obstacle[0], obstacle[1]), obstacle[2], obstacle[3], edgecolor='k', facecolor='gray')
        plt.gca().add_patch(obstacle_rect)

    # Plotting the path
    if len(path) > 1:
        plt.plot([p[0] for p in path], [p[1] for p in path], '-o', label='Path', lw=3, zorder=int(10**4))

    # Plotting the entire tree
    if V_near is not None:
        for node in V_near:
            if node.parent is not None:
                plt.plot([node.point[0], node.parent.point[0]], [node.point[1], node.parent.point[1]], 'k-', lw=0.5)
    elif edges is not None:
        for parent, child in edges:
            plt.plot([parent.point[0], child.point[0]], [parent.point[1], child.point[1]], 'k-', lw=0.5)
    else:
        raise Exception("Must provide either edges or V_near")

    if example == 1:
        save_path = join(PATH, "examples", "two boxes")
    elif example == 2:
        save_path = join(PATH, "examples", "windy maze")
    else:
        raise Exception("Invalid example number")

    # Add grid, legend, and title
    plt.grid(True)
    plt.title(f"{alg_titles[alg_label]} Algorithm. Cost = {cost}")
    plt.legend()
    
    if params is not None:
        plt.savefig(join(save_path, f"{alg_label}_animation_k={params['k']}_eta={params['eta']}.png"), dpi=300)
    else:
        plt.savefig(join(save_path,f"{alg_label}_animation.png"), dpi=300)
    
    plt.show()

def animate_edges_and_path(start, obstacles, goal, path, alg_label, cost, edges=None, V_near=None, params=None, example=1, static_nodes=False):
    alg_titles = {'rrtstar': "RRT٭",
                  'rrt': "RRT",
                  'fmtstar': "FMT٭",
                  'prmstar': "PRM٭"}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis([0, 1, 0, 1])
    
    # Plot start and goal
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    goal_polygon = plt.Polygon(goal, edgecolor='g', facecolor=(0, 1, 0, 0.5), label='Goal')
    ax.add_patch(goal_polygon)

    # Plot obstacles
    for obstacle in obstacles:
        obstacle_rect = plt.Rectangle((obstacle[0], obstacle[1]), obstacle[2], obstacle[3], edgecolor='k', facecolor='gray')
        ax.add_patch(obstacle_rect)
    
    def update(frame):
        ax.set_title(f"{alg_titles[alg_label]} Algorithm. Cost = {cost}")
        if edges is not None:
            if static_nodes:
                ax.scatter([node.point[0] for node in V_near], [node.point[1] for node in V_near], facecolors='none', edgecolors='black', 
                            s=10, label='Sampled Points' if frame == 0 else "")
            # Animate edges
            if frame < len(edges):
                parent, child = edges[frame]
                ax.plot([parent.point[0], child.point[0]], [parent.point[1], child.point[1]], 'k-', lw=0.5)
            # Display path
            else:
                if len(path) > 1:
                    ax.plot([p[0] for p in path], [p[1] for p in path], '-o', color='blue', lw=3, label='Path' if frame == len(edges) else "")
                    if frame == len(edges):
                        ax.legend()
        elif V_near is not None:
            if frame < len(V_near):
                node = V_near[frame]
                if node.parent is not None:
                    ax.plot([node.point[0], node.parent.point[0]], [node.point[1], node.parent.point[1]], 'k-', lw=0.5)
            # Display path
            else:
                if len(path) > 1:
                    ax.plot([p[0] for p in path], [p[1] for p in path], '-o', lw=3, label='Path' if frame == len(V_near) else "")
                    if frame == len(V_near):
                        ax.legend()
        else:
            raise Exception("Must provide either edges or V_near")

        return ax
    
    frame_length = len(edges) if edges is not None else len(V_near)

    ani = animation.FuncAnimation(fig, update, frames=frame_length + 10, blit=False, repeat=False, interval=1)

    if example == 1:
        save_path = join(PATH, "examples", "two boxes")
    elif example == 2:
        save_path = join(PATH, "examples", "windy maze")
    else:
        raise Exception("Invalid example number")

    if params is not None:
        ani.save(join(save_path, f"{alg_label}_animation_k={params['k']}_eta={params['eta']}.mp4"), writer='ffmpeg', fps=60)
    else:
        ani.save(join(save_path,f"{alg_label}_animation.mp4"), writer='ffmpeg', fps=60)
    
    ax.grid(False)
    plt.show()

def animate_edges_and_path_HTML(start, obstacles, goal, path, alg_label, cost, edges=None, V_near=None, params=None, example=1, static_nodes=False):
    alg_titles = {'rrtstar': "RRT٭",
                  'rrt': "RRT",
                  'fmtstar': "FMT٭",
                  'prmstar': "PRM٭"}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis([0, 1, 0, 1])
    
    # Plot start and goal
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    goal_polygon = plt.Polygon(goal, edgecolor='g', facecolor=(0, 1, 0, 0.5), label='Goal')
    ax.add_patch(goal_polygon)

    # Plot obstacles
    for obstacle in obstacles:
        obstacle_rect = plt.Rectangle((obstacle[0], obstacle[1]), obstacle[2], obstacle[3], edgecolor='k', facecolor='gray')
        ax.add_patch(obstacle_rect)
    
    def update(frame):
        ax.set_title(f"{alg_titles[alg_label]} Algorithm. Cost = {cost}")
        if static_nodes:
            ax.scatter([node.point[0] for node in V_near], [node.point[1] for node in V_near], facecolors='none', edgecolors='black', 
                        s=10, label='Sampled Points' if frame == 0 else "")
        if edges is not None:
            # Animate edges
            if frame < len(edges):
                parent, child = edges[frame]
                ax.plot([parent.point[0], child.point[0]], [parent.point[1], child.point[1]], 'k-', lw=0.5)
            # Always display path in frames after the tree is fully animated
            if frame >= len(edges) and len(path) > 1:
                ax.plot([p[0] for p in path], [p[1] for p in path], '-o', color='blue', lw=3, label='Path' if frame == len(edges) else "")
                if frame == len(edges):
                    ax.legend()
        elif V_near is not None:
            if frame < len(V_near):
                node = V_near[frame]
                if node.parent is not None:
                    ax.plot([node.point[0], node.parent.point[0]], [node.point[1], node.parent.point[1]], 'k-', lw=0.5)
            # Always display path in frames after the tree is fully animated
            if frame >= len(V_near) and len(path) > 1:
                ax.plot([p[0] for p in path], [p[1] for p in path], '-o', lw=3, label='Path' if frame == len(V_near) else "")
                if frame == len(V_near):
                    ax.legend()
        else:
            raise Exception("Must provide either edges or V_near")

        return ax
    
    frame_length = len(edges) if edges is not None else len(V_near)

    ani = animation.FuncAnimation(fig, update, frames=frame_length + 10, blit=False, repeat=False, interval=100)

    plt.close(fig)  # Close the figure to prevent it from displaying in the output
    return HTML(ani.to_html5_video())

def generateValidationSamples(obstacles, Nmax, seed, path=None):
    # Seed the random number generator
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random samples
    rand_samples = [(np.random.uniform(0, 1), np.random.uniform(0, 1)) for _ in range(Nmax)]
    collision_free_samples = [sample for sample in rand_samples if not is_point_inside_obstacle(sample, obstacles)]
    
    # Write to file
    if path is not None:
        pathTofile = join(path, "validation_samples.csv")
    else:
        pathTofile = "validation_samples.csv"
    with open(pathTofile, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(collision_free_samples)

def extract_edges_and_nodes(V, A):
    edges = []
    for idx, node in enumerate(V):
        if A[idx] is not None:  # Check if the node has a parent
            edges.append((V[A[idx]].point, node.point))  # Add edge as (parent, current_node)
    return edges, [node.point for node in V]


def animate_rrt_star(start, goal, obstacles, path, cost, params, example=1):
    added_edges = params['added_edges']
    rewired_edges = params['rewired_edges']
    lengthFrames = len(added_edges)

    new_edges_for_rewired = {}
    for old_edge in rewired_edges:
        child_node = old_edge[1]
        new_edge = next((edge for edge in added_edges if edge[1] == child_node and edge != old_edge), None)
        if new_edge:
            new_edges_for_rewired[old_edge] = new_edge

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis([0, 1, 0, 1])
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    goal_polygon = plt.Polygon(goal, edgecolor='g', facecolor=(0, 1, 0, 0.5), label='Goal')
    ax.add_patch(goal_polygon)

    for obstacle in obstacles:
        obstacle_rect = plt.Rectangle((obstacle[0], obstacle[1]), obstacle[2], obstacle[3], edgecolor='k', facecolor='gray')
        ax.add_patch(obstacle_rect)

    edge_to_line = {}
    drawn_nodes = set([tuple(start)])
    child_to_parent = {}

    def update(frame):
        ax.set_title(f"RRT* Algorithm - Cost: {cost}")

        if frame < len(added_edges):
            parent, child = added_edges[frame]

            # If this child was previously drawn (due to an old edge), remove that line
            if tuple(child) in child_to_parent:
                old_parent = child_to_parent[tuple(child)]
                old_edge = edge_to_line.get((old_parent, tuple(child)))
                if old_edge:
                    old_edge.remove()
                    del edge_to_line[(old_parent, tuple(child))]

            # Draw the edge
            line, = ax.plot([parent[0], child[0]], [parent[1], child[1]], 'k-', lw=0.5)
            edge_to_line[(tuple(parent), tuple(child))] = line

            # Update child-to-parent mapping and drawn_nodes set
            child_to_parent[tuple(child)] = tuple(parent)
            drawn_nodes.add(tuple(child))

        if frame == len(added_edges) and len(path) > 1:
            ax.plot([p[0] for p in path], [p[1] for p in path], '-o', lw=3, label='Path')
            ax.legend()

        return ax

    PATH = "."  # Adjust this to your desired path
    if example == 1:
        save_path = join(PATH, "examples", "two boxes")
    elif example == 2:
        save_path = join(PATH, "examples", "windy maze")
    else:
        raise Exception("Invalid example number")

    interval = 1000 / 30  # for 30 fps
    ani = FuncAnimation(fig, update, frames=lengthFrames + 10, blit=False, repeat=False, interval=interval)
    ani.save(join(save_path, f"rrtstar_animation.mp4"), writer='ffmpeg', fps=30)
    plt.show()

def animate_rrt_star_HTML(start, goal, obstacles, path, cost, params, example=1):
    added_edges = params['added_edges']
    rewired_edges = params['rewired_edges']
    lengthFrames = len(added_edges)

    new_edges_for_rewired = {}
    for old_edge in rewired_edges:
        child_node = old_edge[1]
        new_edge = next((edge for edge in added_edges if edge[1] == child_node and edge != old_edge), None)
        if new_edge:
            new_edges_for_rewired[old_edge] = new_edge

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis([0, 1, 0, 1])
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    goal_polygon = plt.Polygon(goal, edgecolor='g', facecolor=(0, 1, 0, 0.5), label='Goal')
    ax.add_patch(goal_polygon)

    for obstacle in obstacles:
        obstacle_rect = plt.Rectangle((obstacle[0], obstacle[1]), obstacle[2], obstacle[3], edgecolor='k', facecolor='gray')
        ax.add_patch(obstacle_rect)

    edge_to_line = {}
    drawn_nodes = set([tuple(start)])
    child_to_parent = {}

    def update(frame):
        ax.set_title(f"RRT* Algorithm - Cost: {cost}")

        if frame < len(added_edges):
            parent, child = added_edges[frame]

            # If this child was previously drawn (due to an old edge), remove that line
            if tuple(child) in child_to_parent:
                old_parent = child_to_parent[tuple(child)]
                old_edge = edge_to_line.get((old_parent, tuple(child)))
                if old_edge:
                    old_edge.remove()
                    del edge_to_line[(old_parent, tuple(child))]

            # Draw the edge
            line, = ax.plot([parent[0], child[0]], [parent[1], child[1]], 'k-', lw=0.5)
            edge_to_line[(tuple(parent), tuple(child))] = line

            # Update child-to-parent mapping and drawn_nodes set
            child_to_parent[tuple(child)] = tuple(parent)
            drawn_nodes.add(tuple(child))

        if frame == len(added_edges) and len(path) > 1:
            ax.plot([p[0] for p in path], [p[1] for p in path], '-o', lw=3, label='Path')
            ax.legend()

        return ax

    PATH = "."  # Adjust this to your desired path
    if example == 1:
        save_path = join(PATH, "examples", "two boxes")
    elif example == 2:
        save_path = join(PATH, "examples", "windy maze")
    else:
        raise Exception("Invalid example number")

    interval = 1000 / 30  # for 30 fps
    ani = FuncAnimation(fig, update, frames=lengthFrames + 10, blit=False, repeat=False, interval=interval)
    
    plt.close(fig)  # Close the figure to prevent it from displaying in the output
    return HTML(ani.to_html5_video())

def visualize_prm_star(start, goal, obstacles, path, edges, cost, V):
    """
    Visualizes the PRM* graph, obstacles, start, goal, and the computed path.

    Parameters:
    - start: Tuple of start coordinates (x, y)
    - goal: List of goal polygon coordinates [(x1, y1), (x2, y2), ...]
    - obstacles: List of obstacles, each defined by a list of coordinates [(x1, y1), (x2, y2), ...]
    - path: List of path points [(x1, y1), (x2, y2), ...]
    - edges: List of graph edges [(Node1, Node2), ...]
    - cost: Total cost of the shortest path
    - V: List of nodes
    """
    
    fig, ax = plt.subplots()
    
    # Plot the start and goal
    ax.axis([0, 1, 0, 1])
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    goal_polygon = plt.Polygon(goal, edgecolor='g', facecolor=(0, 1, 0, 0.5), label='Goal')
    ax.add_patch(goal_polygon)

    for obstacle in obstacles:
        obstacle_rect = plt.Rectangle((obstacle[0], obstacle[1]), obstacle[2], obstacle[3], edgecolor='k', facecolor='gray')
        ax.add_patch(obstacle_rect)
        
    # Plot the nodes
    for node in V:
        ax.plot(node.point[0], node.point[1], 'o', color='none', markeredgecolor='k', markersize=5)
        
    # Plot the PRM* graph
    for edge in edges:
        x_vals, y_vals = zip(*[edge[0].point, edge[1].point])
        ax.plot(x_vals, y_vals, 'k-', lw=0.5)  # edges as black lines
    
    # Plot the path
    path_x, path_y = zip(*path)
    ax.plot(path_x, path_y, 'b-', linewidth=2)  # path as thick blue line
    
    # Setting plot title with cost and showing the plot
    ax.set_title(f"PRM* Path Planning (Cost: {cost})")
    plt.axis('equal')
    plt.grid(True)
    plt.show()