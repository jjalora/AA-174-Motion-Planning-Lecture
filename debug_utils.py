import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point

def plot_samples_and_obstacles(V, obstacles, goal):
    plt.figure(figsize=(10, 10))

    # Plotting the obstacles
    for obstacle in obstacles:
        # Convert rectangle [x, y, width, height] to polygon vertices
        x, y, width, height = obstacle
        obstacle_polygon = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        xs, ys = zip(*obstacle_polygon)  # Unzip the coordinates
        plt.plot(xs + (xs[0],), ys + (ys[0],), 'k-')  # Plot the obstacle with black line

    # Plotting the goal
    x, y = zip(*goal)  # Unzip the coordinates
    plt.plot(x + (x[0],), y + (y[0],), 'g-')  # Plot the goal with green line

    # Plotting the sampled points
    x_samples = [node.point[0] for node in V]
    y_samples = [node.point[1] for node in V]
    plt.scatter(x_samples, y_samples, c='b', s=10, label='Sampled Points')  # Blue points

    # Additional plot settings
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sampled Points and Obstacles')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Equal scaling
    plt.show()  # Display the plot