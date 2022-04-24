import numpy as np
from coverage_planner import CoveragePlanner, HeuristicType, PlannerStatus
from tabulate import tabulate

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

cp_debug_level = 0
test_show_each_result = False

# Load map
def load_map(map_name):
    with open("maps/{}.npy".format(map_name), 'rb') as f:
        return np.load(f)

# Plot the results using matplotlib
def plot_map(target_map, trajectory, map_name="map", params_str=""):
    # References from CoveragePlanner to convert actions to oriented movements
    movement = [[-1,  0],  # up
                [0, -1],    # left
                [1,  0],    # down
                [0,  1]]    # right
    action = [-1, 0, 1, 2]

    # Create a figure
    fig, ax = plt.subplots()

    # Define the colors
    start_position_color = 'gold'
    start_orientation_color = 'deeppink'
    status_color_ref = {PlannerStatus.STANDBY: 'black',
                        PlannerStatus.COVERAGE_SEARCH: 'royalblue',
                        PlannerStatus.NEARST_UNVISITED_SEARCH: 'darkturquoise',
                        PlannerStatus.FOUND: 'mediumseagreen',
                        PlannerStatus.NOT_FOUND: 'red'}
    cmap = mpl.colors.ListedColormap(
        ['w', 'k', start_position_color, status_color_ref[PlannerStatus.FOUND], status_color_ref[PlannerStatus.NOT_FOUND]])
    norm = mpl.colors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

    # Define conversions from status to cmap references idx.
    status_to_cmap_pos = {PlannerStatus.FOUND: 3,
                          PlannerStatus.NOT_FOUND: 4}

    # Copy the original map to avoid changes on it
    target_map_ref = np.copy(target_map)

    # Add to the map a reference of the last visited position, which will reflect its color on the map
    target_map_ref[
        trajectory[-1][1]][trajectory[-1][2]] = status_to_cmap_pos[trajectory[-1][6]]

    # plot the colored map
    ax.imshow(target_map_ref, interpolation='none', cmap=cmap, norm=norm)

    # Plot an arrow on each action of the trajectory
    for i in range(len(trajectory)-1):

        x = trajectory[i][2]
        y = trajectory[i][1]

        # Add the action value to the current orientation will result in the movement index
        mov_idx = (trajectory[i][3]+action[trajectory[i][5]]) % len(movement)
        mov = movement[mov_idx]

        # Get the correspondent status color from the reference list
        arrow_color = status_color_ref[trajectory[i][6]]

        # Just to improve visualization, translate A* arrows slight to right/down
        if trajectory[i][6] == PlannerStatus.NEARST_UNVISITED_SEARCH:
            # Check if movement is vertical or horizontal
            if mov_idx % 2:
                y -= 0.25
            else:
                x += 0.25

        # Add the arrow point from the current position to the next position
        ax.arrow(x, y, mov[1], mov[0], width=0.1,
                 color=arrow_color, length_includes_head=True)

    # Plot the inital orientation
    init_direction = np.array(movement[trajectory[0][3]])/2
    ax.arrow(trajectory[0][2]-init_direction[1]/2, trajectory[0][1]-init_direction[0]/2, init_direction[1], init_direction[0], width=0.1,
             color=start_orientation_color, length_includes_head=True, head_length=0.2)

    # Add legend
    legend_elements = [Line2D([0], [0], color=status_color_ref[PlannerStatus.COVERAGE_SEARCH], lw=1, marker='>',
                              markerfacecolor=status_color_ref[PlannerStatus.COVERAGE_SEARCH], label='Coverage Search'),
                       Line2D([0], [0], color=status_color_ref[PlannerStatus.NEARST_UNVISITED_SEARCH], lw=1, marker='>',
                              markerfacecolor=status_color_ref[PlannerStatus.NEARST_UNVISITED_SEARCH], label='A*u Search'),
                       Line2D([0], [0], color='w', lw=1, marker='>',
                              markerfacecolor=start_orientation_color, label='Start Orientation'),
                       Line2D([0], [0], marker='s', color='w', label='Start Pos',
                              markerfacecolor=start_position_color, markersize=15),
                       Line2D([0], [0], marker='s', color='w', label='End Pos',
                              markerfacecolor=status_color_ref[trajectory[-1][6]], markersize=15),
                       Line2D([0], [0], marker='s', color='w',
                              label='Obstacle', markerfacecolor='k', markersize=15),
                       ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(
        1.025, 1.0), loc='upper left')
    plt.title("Coverage Path Planning - {}\n{}".format(map_name, params_str))
    plt.tight_layout()
    plt.show()
    fig.savefig("output_images/{}.png".format(map_name), bbox_inches='tight')


if __name__ == "__main__":
    # Create a list for dynamic compute the best coverage heuristic for each map
    maps = ["map1", "map2", "map3", "map4"]
    cp_heuristics = [HeuristicType.VERTICAL,
                    HeuristicType.HORIZONTAL, HeuristicType.CHEBYSHEV, HeuristicType.MANHATTAN]
    orientations = [0, 1, 2, 3]

    for map_name in maps:
        compare_tb = []

        target_map = load_map(map_name)
        cp = CoveragePlanner(target_map)
        cp.set_debug_level(cp_debug_level)

        # Iterate over each orientation with each heuristic
        for heuristic in cp_heuristics:
            for orientation in orientations:
                if test_show_each_result:
                    print("\n\nIteration[map:{}, cp:{}, initial_orientation:{}]".format(
                        map_name, heuristic.name, orientation))

                cp.start(initial_orientation=orientation, cp_heuristic=heuristic)
                cp.compute()

                if test_show_each_result:
                    cp.show_results()

                res = [heuristic.name, orientation]
                res.extend(cp.result())
                compare_tb.append(res)

        # Sort by number of steps
        compare_tb.sort(key=lambda x: (x[3], x[4]))

        # Show results
        print("Map tested: {}".format(map_name))

        # Print the summary of results for the given map
        summary = [row[0:5] for row in compare_tb]
        for row in summary:
            # Format cost to 2 decimal
            row[4] = "{:.2f}".format(row[4])
            # Convert movement index to movement names
            row[1] = cp.movement_name[row[1]]

        compare_tb_headers = ["Heuristic",
                            "Orientation", "Found?", "Steps", "Cost"]
        summary_tb = tabulate(summary, compare_tb_headers,
                            tablefmt="pretty", floatfmt=".2f")
        print(summary_tb)

        # Print the policy map of the best coverage planner
        cp.print_policy_map(trajectory=compare_tb[0][5], trajectory_annotations=[])

        # Plot the complete trajectory map
        plot_map(target_map, compare_tb[0][5], map_name=map_name,
                params_str="Heuristic:{}, Initial Orientation: {}".format(compare_tb[0][0], cp.movement_name[compare_tb[0][1]]))

        # Print the best path
        print("\nList of coordinates of the best path: [map:{}, initial orientation: {} ({}), coverage path Heuristic:{}]".format(
            map_name, cp.movement_name[compare_tb[0][1]], compare_tb[0][1], compare_tb[0][0]))
        print(compare_tb[0][6])
        print("\n\n")
