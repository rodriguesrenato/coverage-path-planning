import numpy as np
from coverage_planner import CoveragePlanner, HeuristicType

# Load map
def load_map(map_name):
    with open("maps/{}.npy".format(map_name), 'rb') as f:
        return np.load(f)

# Create a list for dynamic compute the best coverage heuristic for each map
maps = ["map1","map2","map3"]
cp_heuristics = [HeuristicType.VERTICAL,
                 HeuristicType.HORIZONTAL, HeuristicType.CHEBYSHEV, HeuristicType.MANHATTAN]


for map_name in maps:
    compare_tb = []

    target_map = load_map(map_name)
    cp = CoveragePlanner(target_map)
    cp.set_debug_level(0)

    for heuristic in cp_heuristics:
        print("\n\nIteration[map:{}, cp:{}]".format(map_name,heuristic.name))
        cp.start(cp_heuristic=heuristic)
        cp.compute()
        cp.show_results()
        res = cp.result()
        res.insert(0, heuristic.name)
        compare_tb.append(res)

    # Sort by number of steps
    compare_tb.sort(key=lambda x: x[2])

    print("\nmap tested: {}".format(map_name))
    print("CP_Heur.\tFound?\tSteps\tCost")
    for c in compare_tb:
        print("{}\t{}\t{}\t{:.2f}".format(c[0],c[1],c[2],c[3]))

    print("\nList of coordinates of the best pathÇ [map:{}, cp:{}]".format(map_name,heuristic.name))
    print(compare_tb[0][5])
