import numpy as np
import copy
from enum import Enum, IntEnum, auto
np.set_printoptions(precision=2)


class PlannerStatus(Enum):
    RESIGN = auto()
    SEARCH = auto()
    FOUND = auto()
    NOT_FOUND = auto()
    START = auto()
    COVERAGE_SEARCH = auto()
    NEARST_UNVISITED_SEARCH = auto()
    STANDBY = auto()


class Robot():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.orientation = 0
        self.action = 0


class CoveragePlanner():

    def __init__(self, map_open):
        self.grid = map_open
        self.heuristic = map_open
        self.init = self.GetStartPoint(map_open)
        self.goal = self.init
        self.delta = [[-1, 0],  # go up
                      [0, -1],  # go left
                      [1, 0],  # go down
                      [0, 1]]  # go right
        self.cost = [1, 1, 1, 1]
        self.delta_name = ['^', '<', 'v', '>']
        self.delta_orientation = {'^': 0, '<': 1, 'v': 2, '>': 3}

        # Second approach, holding robots orientation
        self.forward = [[-1,  0],  # go up
                        [0, -1],  # go left
                        [1,  0],  # go down
                        [0,  1]]  # go right
        # ['up', 'left', 'down', 'right']
        self.forward_name = ['^', '<', 'v', '>']

        self.action = [-1, 0, 1, 2]
        self.action_name = ['R', '#', 'L', 'B']  # Right,Forward,Left,Backwards
        self.action_cost = [.2, .1, .2, .4]

        self.current_x = self.init[0]
        self.current_y = self.init[1]
        self.current_o = 0
        self.current_pos = [self.init[0], self.init[1], 0]

        self.current_trajectory = []
        self.current_trajectory_annotations = []

        self.coverage_grid = np.copy(map_open)
        self.state_ = PlannerStatus.STANDBY


        # res =  [Success?, total_cost_actions, trajectory, final_coverage_grid]
        # trajectory = [value , x, y, orientation, action_performed_to_get_here, next_action]

    def compute(self):

        searching = True
        if self.state_ == PlannerStatus.START:
            print("[compute] {}".format(self.state_.name))
            # Restart local vars and prepare for a clean start searching

            self.current_x = self.init[0]
            self.current_y = self.init[1]
            self.current_o = 0
            self.coverage_grid = np.copy(self.grid)
            self.current_trajectory = []

            self.state_ = PlannerStatus.COVERAGE_SEARCH

        elif self.state_ == PlannerStatus.COVERAGE_SEARCH:
            print("[compute] {}".format(self.state_.name))

            # Search using coverage_search algorithm
            res = self.coverage_search(
                [self.current_x, self.current_y, self.current_o])

            # Update the current position to the final search position
            self.current_x = res[2][-1][1]
            self.current_y = res[2][-1][2]
            self.current_o = res[2][-1][3]

            # If already has trajectory positions, remove duplicated positions by joining actions and add a special obs at the end of the new first position list
            if len(self.current_trajectory) >0 and len(res[2])>0:
                # prev_action = self.current_trajectory[-1][4]
                # res[2][0][4] = prev_action
                res[2][0][4] = self.current_trajectory[-1][4]
                self.current_trajectory_annotations.append([res[2][0][1],res[2][0][2],"CS"])
                self.current_trajectory.pop()

            # Add the computed path to the trajectory list
            for t in res[2]:
                self.current_trajectory.append(t)

            # Update the current coverage_grid
            self.coverage_grid = res[3]

            if res[0]:
                self.state_ = PlannerStatus.FOUND
            else:
                self.state_ = PlannerStatus.NEARST_UNVISITED_SEARCH

            print("[compute] {} - Finished at {},{},{} - steps: {}".format(self.state_.name,
                  self.current_x, self.current_y, self.current_o, len(self.current_trajectory)))

        elif self.state_ == PlannerStatus.NEARST_UNVISITED_SEARCH:
            print("[compute] {}".format(self.state_.name))

            # Search using a_star_search_closest_unvisited algorithm
            res = self.a_star_search_closest_unvisited(
                [self.current_x, self.current_y, self.current_o])

            if res[0]:
                # Update the current position to the final search position
                self.current_x = res[2][-1][1]
                self.current_y = res[2][-1][2]
                self.current_o = res[2][-1][3]

                # If already has trajectory positions, remove duplicated positions by joining actions and add a special obs at the end of the new first position list
                if len(self.current_trajectory) >0 and len(res[2])>0:
                    # prev_action = self.current_trajectory[-1][4]
                    # res[2][0][4] = prev_action
                    res[2][0][4] = self.current_trajectory[-1][4]
                    self.current_trajectory_annotations.append([res[2][0][1],res[2][0][2],"A*"])
                    self.current_trajectory.pop()

                # Add the computed path to the trajectory list
                for t in res[2]:
                    self.current_trajectory.append(t)

                self.state_ = PlannerStatus.COVERAGE_SEARCH
            else:
                self.state_ = PlannerStatus.NOT_FOUND

        elif self.state_ == PlannerStatus.FOUND:
            print("[compute] {}".format(self.state_.name))

            print("\n[coverage_search] trajectory: ")
            for t in self.current_trajectory:
                print("{:.2f}\t{}\t{}\t{}\t{}\t{}".format(
                    t[0], t[1], t[2], t[3], t[4], t[5]))
            self.PrintPolicyMap()
            searching = False

        elif self.state_ == PlannerStatus.NOT_FOUND:
            print("[compute] {}".format(self.state_.name))
            self.PrintPolicyMap()
            searching = False
        else:
            print("Invalid State")
            searching = False
        return searching

    def start(self):
        self.state_ = PlannerStatus.START

    def coverage_search(self, initial_pos):
        # Create a reference grid for visited coords
        closed = np.copy(self.coverage_grid)
        closed[initial_pos[0]][initial_pos[1]] = 1

        print("[coverage_search] initial closed: ")
        print(closed)

        policy = [[" " for row in range(len(self.grid[0]))]
                  for col in range(len(self.grid))]

        for col in range(len(self.grid[0])):
            for row in range(len(self.grid)):
                if self.grid[row][col] == 1:
                    policy[row][col] = "XXX"

        # Define the heuristic matrix TODO: PARAMETERIZE
        heuristic = self.CreateVerticalHeuristic(initial_pos)
        print("[coverage_search] Heuristic: ")
        print(heuristic)

        x = initial_pos[0]
        y = initial_pos[1]
        o = initial_pos[2]
        v = 0

        # Fill the initial coord in the iteration list
        trajectory_coords = [[v, x, y, o, None, None]]
        policy[trajectory_coords[-1][1]][trajectory_coords[-1][2]] = "STA"
        complete_coverage = False
        resign = False

        while not complete_coverage and not resign:

            if self.check_full_coverage2(self.grid, closed):
                print("Complete coverage")
                complete_coverage = True
                policy[trajectory_coords[-1][1]
                       ][trajectory_coords[-1][2]] = "END"

            else:
                # Get the last visited coord info
                v = trajectory_coords[-1][0]
                x = trajectory_coords[-1][1]
                y = trajectory_coords[-1][2]
                o = trajectory_coords[-1][3]

                # [accumulated_cost, x_pos, y_pos, orientation, action_performed_to_get_here, next_action]
                possible_next_coords = []

                # calculate the possible next coords
                for a in range(len(self.action)):
                    o2 = (self.action[a]+o) % len(self.forward)
                    x2 = x + self.forward[o2][0]
                    y2 = y + self.forward[o2][1]
                    # a2 = a

                    if x2 >= 0 and x2 < len(self.grid) and y2 >= 0 and y2 < len(self.grid[0]):
                        if closed[x2][y2] == 0 and self.grid[x2][y2] == 0:
                            v2 = v + self.action_cost[a] + heuristic[x2][y2]
                            possible_next_coords.append(
                                [v2, x2, y2, o2, a, None])

                # If there isn any possible next position, stop searching
                if len(possible_next_coords) == 0:
                    resign = True
                    print("Could not find a next unvisited coord")

                    policy[x][y] = "*"

                # otherwise update the trajectory list wiht the next position with the lowest possible cost
                else:
                    # rank by total_cost
                    possible_next_coords.sort(key=lambda x: x[0])

                    # update the last trajectory_coords next_action
                    trajectory_coords[-1][5] = possible_next_coords[0][4]

                    # add the lowest cost possible_next_coords to the trajectory_coords list
                    trajectory_coords.append(possible_next_coords[0])

                    # mark the chosen possible_next_coords position as visited
                    closed[possible_next_coords[0][1]
                           ][possible_next_coords[0][2]] = 1

                    # self.coverage_grid[possible_next_coords[0][1]][possible_next_coords[0][2]] = 1

                    policy[x][y] += self.forward_name[o] + \
                        self.action_name[possible_next_coords[0][4]]

        print("\n[coverage_search] closed: ")
        print(closed)

        print("\n[coverage_search] policy: ")
        self.PrintMap(policy)

        print("\n[coverage_search] trajectory: ")
        for t in trajectory_coords:
            print("{:.2f}\t{}\t{}\t{}\t{}\t{}".format(
                t[0], t[1], t[2], t[3], t[4], t[5]))

        # Calculate total cost
        total_cost = 0
        for i in range(len(trajectory_coords)):
            action = trajectory_coords[i][5]
            if action is not None:
                total_cost += self.action_cost[action]

        print("Steps: {}".format(len(trajectory_coords)))
        # Pack the standard answer
        # [Success?, total_cost_actions, trajectory, final_coverage_grid]
        # trajectory = [value , x, y, orientation, action_performed_to_get_here, next_action]
        res = [not resign, total_cost, trajectory_coords, closed]

        return res

    # Find the shortest path between init and goal coords

    def a_star_search_closest_unvisited(self, initial_pos):
        print("\n[a_star_search_closest_unvisited]\n\n")

        # Create a reference grid for visited coords
        closed = np.zeros_like(self.grid)
        closed[initial_pos[0]][initial_pos[1]] = 1

        print("[a_star_search_closest_unvisited] initial closed: ")
        print(closed)

        expand = np.full((np.size(self.grid, 0), np.size(self.grid, 1)), -1)
        orientation = np.full(
            (np.size(self.grid, 0), np.size(self.grid, 1)), -1)

        # Define the heuristic function that will be used by A*
        heuristic = self.CreateManhattanHeuristic(initial_pos[0:2])
        print("[a_star_search_closest_unvisited] Heuristic: ")
        print(heuristic)

        x = initial_pos[0]
        y = initial_pos[1]
        o = initial_pos[2]
        g = 0
        f = g + heuristic[x][y]

        # trajectory_coords = [[f, x, y, o, orientation]]
        open = [[f, g, x, y]]

        found = False  # Whether a unvisited position is found or not
        resign = False  # flag set if we can't find expand
        count = 0

        x2 = initial_pos[0]
        y2 = initial_pos[1]

        while not found and not resign:
            print("[a_star_search_closest_unvisited] open: {}".format(open))

            # If there isn't any more positions to expand, the unvisited position could not be found, then resign
            if len(open) == 0:
                resign = True
                print("[a_star_search_closest_unvisited] Fail to find a path")

            # Otherwise expand search again
            else:

                # Sort open positions list by total cost, in descending order and reverse it to pop the element with lowest total cost
                open.sort(key=lambda x: x[0])  # +heuristic[x[1]][x[2]])
                open.reverse()
                next = open.pop()

                # update current search x,y,g and set
                x = next[2]
                y = next[3]
                g = next[1]
                expand[x][y] = count
                count += 1

                # Check if a unvisited position is found
                if self.coverage_grid[x][y] == 0:
                    found = True
                    x2 = x
                    y2 = y
                else:
                    # calculate the possible next coords
                    for i in range(len(self.forward)):
                        x2 = x + self.forward[i][0]
                        y2 = y + self.forward[i][1]

                        # Check if it is out of bounds or already visited
                        if x2 >= 0 and x2 < len(self.grid) and y2 >= 0 and y2 < len(self.grid[0]):
                            if closed[x2][y2] == 0 and self.grid[x2][y2] == 0:
                                g2 = g + self.cost[i]
                                f = g2 + heuristic[x2][y2]
                                open.append([f, g2, x2, y2])
                                closed[x2][y2] = 1
                                orientation[x2][y2] = i

        print("\n[a_star_search_closest_unvisited] expand: ")
        print(expand)

        print("\n[a_star_search_closest_unvisited] orientation: ")
        print(orientation)

        trajectory = []
        total_cost = 0

        # If path was found, then build the trajectory
        if found:

            policy = np.full(
                (np.size(self.grid, 0), np.size(self.grid, 1)), " ")

            # x2 and y2 are the unvisited position
            x = x2
            y = y2
            policy[x][y] = '*'

            trajectory = [[0, x, y, orientation[x][y], None, None, "EA*"]]
            orientation[initial_pos[0]][initial_pos[1]]=initial_pos[2]

            # Go backwards from the unvisited position founded to this search inital position
            while (x != initial_pos[0] or y != initial_pos[1]):
                # Calculate the path predecessor position
                x0 = x - self.delta[orientation[x][y]][0]
                y0 = y - self.delta[orientation[x][y]][1]
                o0 = orientation[x0][y0]
                a0 = None

                # Compute the required action index to get from the predecessor position to current iteration position
                a = (trajectory[-1][3]-o0 + 1) % len(self.action)

                policy[x0][y0] = self.delta_name[orientation[x][y]]

                # Update the successor position "action_performed_to_get_here"
                trajectory[-1][4] = a

                # Add the path predecessor position
                trajectory.append([0, x0, y0, o0, a0, a])
                # trajectory.append([0, x0, y0, o0, a0, a, self.action_name[a], self.delta_name[orientation[x][y]]])

                # update x and y for next iteration
                x = x0
                y = y0

            trajectory.reverse()

            print("\n[a_star_search_closest_unvisited] policy: ")
            print(policy)

            print("\n[a_star_search_closest_unvisited] trajectory: ")
            for t in trajectory:
                print(t)

        print("Steps: {}".format(len(trajectory)))

        # Pack the standard answer
        # trajectory:   [value , x, y, orientation, action_performed_to_get_here, next_action]
        # res:          [Success?, total_cost_actions, trajectory, final_coverage_grid]
        res = [found, total_cost, trajectory, None]

        return res

    def check_full_coverage(self):
        return np.all(self.coverage_grid)

    def check_full_coverage2(self, grid, closed):
        return np.all(np.copy(grid)+np.copy(closed))

    def CreateManhattanHeuristic(self, target_point):
        heuristic = np.zeros_like(self.grid)
        for x in range(len(heuristic)):
            for y in range(len(heuristic[0])):
                heuristic[x][y] = abs(x-target_point[0])+abs(y-target_point[1])
        return heuristic

    def CreateChebyshevHeuristic(self, target_point):
        heuristic = np.zeros_like(self.grid)
        for x in range(len(heuristic)):
            for y in range(len(heuristic[0])):
                heuristic[x][y] = max(
                    abs(x-target_point[0]), abs(y-target_point[1]))
        return heuristic

    def CreateHorizontalHeuristic(self, target_point):
        heuristic = np.zeros_like(self.grid)
        for x in range(len(heuristic)):
            for y in range(len(heuristic[0])):
                heuristic[x][y] = abs(x-target_point[0])
        return heuristic

    def CreateVerticalHeuristic(self, target_point):
        heuristic = np.zeros_like(self.grid)
        for x in range(len(heuristic)):
            for y in range(len(heuristic[0])):
                heuristic[x][y] = abs(y-target_point[1])
        return heuristic

    def GetStartPoint(self, m):
        for x in range(len(m)):
            for y in range(len(m[0])):
                if(m[x][y] == 2):
                    return [x, y]

    def PrintMap(self, m):
        for row in m:
            s = "["
            for i in range(len(m[0])):
                if type(row[i]) is str:
                    s += row[i]
                else:
                    s += "{:.1f}".format(row[i])
                if i is not (len(m[0])-1):
                    s += "\t,"
                else:
                    s += "\t]"
            print(s)

    def Print(self):
        self.PrintMap(self.expand)

    def PrintPolicyMap(self):
        policy = [[" " for row in range(len(self.grid[0]))]
                  for col in range(len(self.grid))]

        for col in range(len(self.grid[0])):
            for row in range(len(self.grid)):
                if self.grid[row][col] == 1:
                    policy[row][col] = "XXXXXX"

        for t in self.current_trajectory:
            if t[5] is not None:
                policy[t[1]][t[2]] += self.action_name[t[5]]
        
        self.current_trajectory_annotations.append([self.current_trajectory[0][1],self.current_trajectory[0][2],"STA"])
        self.current_trajectory_annotations.append([self.current_trajectory[-1][1],self.current_trajectory[-1][2],"END"])

        for t in self.current_trajectory_annotations:
            policy[t[0]][t[1]] += "@"+t[2]
        
        self.PrintMap(policy)



'''
    def a_star_search(self, init, goal):

        closed = np.zeros_like(self.grid)
        closed[init[0]][init[1]] = 1

        expand = np.full((np.size(self.grid, 0), np.size(self.grid, 1)), -1)
        action = np.full((np.size(self.grid, 0), np.size(self.grid, 1)), -1)

        heuristic = self.CreateManhattanHeuristic(goal)
        print("Heuristic: ")
        print(heuristic)

        x = init[0]
        y = init[1]
        g = 0
        f = g + heuristic[x][y]

        open = [[f, g, x, y]]

        found = False  # flag that is set when search is complete
        resign = False  # flag set if we can't find expand
        count = 0

        x2 = init[0]
        y2 = init[1]

        while not found and not resign:
            if len(open) == 0:
                resign = True
                print("Fail to find a path")
                return None
            else:
                open.sort(key=lambda x: x[0])  # +heuristic[x[1]][x[2]])
                open.reverse()
                next = open.pop()
                x = next[2]
                y = next[3]
                g = next[1]
                expand[x][y] = count
                count += 1

                if x == goal[0] and y == goal[1]:
                    found = True
                else:
                    for i in range(len(self.delta)):
                        x2 = x + self.delta[i][0]
                        y2 = y + self.delta[i][1]
                        if x2 >= 0 and x2 < len(self.grid) and y2 >= 0 and y2 < len(self.grid[0]):
                            if closed[x2][y2] == 0 and self.grid[x2][y2] == 0:
                                g2 = g + self.cost[i]
                                f = g2 + heuristic[x2][y2]
                                open.append([f, g2, x2, y2])
                                closed[x2][y2] = 1
                                action[x2][y2] = i

        print("\nexpand: ")
        print(expand)

        policy = np.full((np.size(self.grid, 0), np.size(self.grid, 1)), " ")
        x = goal[0]
        y = goal[1]
        policy[x][y] = '*'
        
        while (x != init[0] or y != init[1]):
            x2 = x - self.delta[action[x][y]][0]
            y2 = y - self.delta[action[x][y]][1]
            policy[x2][y2] = self.delta_name[action[x][y]]
            x = x2
            y = y2

        print("\npolicy: ")
        print(policy)

        return policy


    def CheckFullCoverage(self):
        for x in range(len(self.map_open)):
            for y in range(len(self.map_open[0])):
                if self.map_open[x][y] == 0 and self.map_closed[x][y] == 0:
                    return False
        return True



    def CreateHeuristic(self):
        self.map_heuristic = np.copy(self.map_open)*-1
        self.PrintMap(self.map_heuristic)



    def a_star_search(self, init, goal):
        closed = np.copy(self.map_open)

    def ExpansionSearch(self):
        self.map_closed = [[0 for col in range(len(self.map_open[0]))]
                           for row in range(len(self.map_open))]
        self.map_closed[self.start_point[0]][self.start_point[1]] = 1

        self.expand = [[-1 for col in range(len(self.map_open[0]))]
                       for row in range(len(self.map_open))]
        self.expand[self.start_point[0]][self.start_point[1]] = 0

        x = self.start_point[0]
        y = self.start_point[1]
        g = 0
        path = [[g, x, y]]
        open_pos = [[g, x, y, path]]

        found = False
        resign = False
        idx = 0
        while found is False and resign is False:
            if len(open_pos) == 0:
                resign = True
                print("Fail to find a path")
                path = "fail"

            else:
                open_pos.sort(key=lambda x: x[0])
                open_pos.reverse
                next_pos = open_pos.pop()

                x = next_pos[1]
                y = next_pos[2]
                g = next_pos[0]
                p = next_pos[3]

                if self.CheckFullCoverage():
                    found = True
                    print("Found path to ({},{}) with g = {}".format(x, y, g))
                    path = p
                else:
                    for i in range(len(self.robot_action)):
                        x2 = x + self.robot_action[i][0]
                        y2 = y + self.robot_action[i][1]

                        if x2 >= 0 and x2 < len(self.map_open) and y2 >= 0 and y2 < len(self.map_open[0]):
                            print("x2:{}\ty2:{}\tlen_self.map_open:{},{}\tlen_self.map_closed:{},{}".format(
                                x2, y2, len(self.map_open), len(self.map_open[0]), len(self.map_closed), len(self.map_closed[0])))
                            # time.sleep(0.1)
                            if self.map_closed[x2][y2] == 0 and self.map_open[x2][y2] == 0:
                                g2 = g + self.robot_action_cost[i]
                                p2 = copy.copy(p)
                                p2.append([g2, x2, y2])
                                open_pos.append([g2, x2, y2, p2])
                                self.map_closed[x2][y2] = 1
                                idx += 1
                                self.expand[x2][y2] = idx

        return found

'''
