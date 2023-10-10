# Motion Planning Algorithms for AA 174A: Principles of Robot Autonomy I

## Overview
This repository contains AA 174 lecture implementations of various motion planning algorithms, such as RRT, RRT*, FMT*, and PRM*, designed to find a path from a starting point to a goal region while avoiding obstacles. The algorithms are implemented in a 2D configuration space and utilize Euclidean distances for simplicity.

## Prerequisites
The code in this repository is written in Python and requires the following libraries:
- `matplotlib`: For plotting and visualizing the paths and obstacles.
- `shapely`: For geometric operations, such as checking if a point is inside a polygon (goal region).
- `numpy`: For numerical operations and handling array data.
- `scipy`: For scientific and technical computing.

## Installation
To install the required libraries, you can use `pip`. First, ensure that you have `pip` installed and then run the following command in your terminal:

bash: ``pip install -r requirements.txt``

## Running the code
The main code for running the examples is located in ``motion_plan.py``. Uncomment the specific algorithm you want to run, then in your terminal
run
``python motion_plan.py``
Ensure that your working directory is set correctly or adjust the file paths accordingly. All plotting functions are located in ``utils.py``.

## Usage
The main functions for each algorithm take the following parameters:
- `start`: A tuple (x, y) representing the starting point of the path.
- `goal`: A list of tuples [(x1, y1), (x2, y2), ...], representing vertices of the polygon defining the goal region.
- `obstacles`: A list of obstacles, each represented by a tuple (x, y, width, height).
- `Nmax`: The maximum number of nodes/samples.
- `eta`: A parameter related to the maximum step size or connection radius (depending on the algorithm).

## Visualization and Plots
All animations and plots are stored and located in the examples directory. Each example has its own dedicated directory. The current examples currently implemented are
- Two Boxes
- Windy Maze

## Questions
Please direct any questions to AA 174A staff or John Alora (jjalora@stanford.edu).