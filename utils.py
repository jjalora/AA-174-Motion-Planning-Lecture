import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point
import numpy as np
from debug_utils import plot_samples_and_obstacles
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

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

def animate_edges_and_path(start, obstacles, goal, path, alg_label, cost, edges=None, V_near=None, params=None, example=1):
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
    
    ax.grid(True)
    plt.show()

def animate_edges_and_path_HTML(start, obstacles, goal, path, alg_label, cost, edges=None, V_near=None, params=None, example=1):
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

    ani = animation.FuncAnimation(fig, update, frames=frame_length + 10, blit=False, repeat=False, interval=1)

    plt.close(fig)  # Close the figure to prevent it from displaying in the output
    return HTML(ani.to_html5_video())