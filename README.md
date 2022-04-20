# coverage-path-planning

A coverage path planning algorithm that combines multiple search algorithms to find a full coverage trajectory with the lowest cost.

Two search algorithm are implemented: "coverage search" and "closest unvisited search", based on "A* search" algorithm. Both algorithm return a standarized message containing:
1. If search completes the full coverage of the map.
2. Total cost of the tracjectory found.
3. The trajectory portion found by the current searchY .
4. The resulted coverage map, which is a matrix that comprises all visisted positions and obstacles.

The trajectory is a list of:

1. Cost at given position
2. X coordinate
3. Y coordinate
4. Orientation (Based on forward actions)
5. Action that was done **before** arrive at this position
6. Action that will be done **after** arrive at this position 

# Challenge

Given a NxM matrix, where free areas are marked with 0, obstacles with 1 and the starting point with 2, find a trajectory that passes through all the free squares with the lowest number of steps. At each step the robot can move to an adjacent (touching sides) free square. The discretisation of the grids is assumed to be equal to the width of the robot that will navigate through the space.

The answer should list the coordinates of the squares it goes through in order from the starting point, essentially the path to be taken by the robot. In addition, the code should include a simple visualisation to verify the results. I've provided you with three areas the algorithm must be able to cope with.

# Stategy

- Understand the problem
- Check for previous solution that I've worked with
    - Udacity nanodegree programs: A* search, dynamic programming, path planning on Self driving Cars.
    - Academic papers of Coverage Path Planning (on `docs` folder)
- Draf the first solution diagram and strategies
- Define first standards (variables, functions response)
- Prototype and test individual approaches
- Evaluate results
- Build the main code structure on a finite state machine
- Test and evaluation
- code clean and optmization after a successfull result
- optimize
- Improve documentation

# Block diagram

```
CoveragePlanner()
    |- compute() #TODO: change to compute_coverage()
    |- start()
    |- coverage_search()
    |- a_star_search_closest_unvisited()
    |- check_full_coverage()
    |- CreateManhattanHeuristic()
    |- CreateChebyshevHeuristic()
    |- CreateHorizontalHeuristic()
    |- CreateVerticalHeuristic()
    |- GetStartPoint()
    |- PrintPolicyMap()
    |- PrintMap()
```

# Results

# References