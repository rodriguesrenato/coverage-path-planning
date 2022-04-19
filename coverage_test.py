import numpy as np
from coverage_planner2 import CoveragePlanner, PlannerStatus

# Load map


def load_map(map_name):
    with open("maps/{}.npy".format(map_name), 'rb') as f:
        return np.load(f)


map_name = "map0"

target_map = load_map(map_name)

print("map to be tested: {}".format(map_name))

cp = CoveragePlanner(target_map)
# cp.PrintMap(target_map)
print("\nStart point: {}".format(cp.init))
cp.start()

while cp.compute():
    print("[main] while compute iteration")
# c = [0, 0, 0, 0]
# i = 0
# Pack the standard answer
# [Success?, total_cost, trajectory, final_coverage_grid]
# trajectory = [value , x, y, orientation, action_taken_on_this_position]

# calculate the cost for each possible start orientation and heuristic
# c[i] = cp.coverage_search(cp.init,i)
# res = cp.coverage_search(cp.init, i, -1)
# trajectory = res[2]
# cp.coverage_grid = res[3]
# # status,trajectory,policy
# if not res[0]:
#     last_coord = [trajectory[-1][1], trajectory[-1][2]]
#     cp.a_star_search_closest_unvisited(last_coord, trajectory[-1][3])

# print("Started at {} heading {}: Total cost = {}".format(
#     cp.init, i, trajectory[-1][0]))
# print("\ntrajectory:")
# cp.PrintMap(trajectory)

# print("\npolicy:")
# cp.PrintMap(policy)

# a_star = cp.a_star_search(cp.init,[0,9])

# print(cp.CreateHorizontalHeuristic(cp.init))
