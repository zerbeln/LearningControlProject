import matplotlib.pyplot as plt
import numpy as np
import csv
from parameters import Parameters as p

def create_learning_curve():
    x_axis = [i for i in range(p.generations+1)]
    y_axis = [0.0 for _ in range(p.generations+1)]
    data_input = np.zeros((p.stat_runs, p.generations+1))

    with open('Output_Data/BestFit.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            for i in range(p.generations+1):
                data_input[line_count, i] = float(row[i])
            line_count += 1

    for i in range(p.generations+1):
        y_axis[i] = np.mean(data_input[:, i])

    plt.figure()
    plt.plot(x_axis, y_axis)
    plt.xlabel('Generations')
    plt.ylabel('Best Fitness')
    plt.savefig('LearningCurve.png')
    # plt.show()

def create_robot_path_plot(srun, walls):

    data_input = [[] for _ in range(p.stat_runs)]
    with open('Output_Data/RobotPath.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            for i in range(p.agent_steps):
                data_input[line_count].append(row[i])
            line_count += 1

    processed_data = []
    for l in data_input[srun]:
        for sub_l in l.split(","):
            sub_l = sub_l.strip('][')
            processed_data.append(sub_l)

    x_coords = np.zeros(p.agent_steps)
    y_coords = np.zeros(p.agent_steps)
    for stp in range(p.agent_steps):
        x_coords[stp] = processed_data[3*stp]
        y_coords[stp] = processed_data[3*stp + 1]

    plt.figure()
    plt.plot(x_coords, y_coords, 'b')
    plt.plot(x_coords[0], y_coords[0], 'bo', markersize=10)
    plt.plot(x_coords[p.agent_steps-1], y_coords[p.agent_steps-1], 'bX', markersize=10)

    # ACTUAL WORLD ---------------------------

    plt.plot((walls[0, 0, 0], walls[0, 1, 0]), (walls[0, 0, 1], walls[0, 1, 1]), 'k-', linewidth=6, label='top wall')
    plt.plot((walls[1, 0, 0], walls[1, 1, 0]), (walls[1, 0, 1], walls[1, 1, 1]), 'k-', linewidth=6, label='bottom wall')

    # CHONK WORLD ----------------------------
    # plt.xlim([0, 30])
    # plt.ylim([0, 30])
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.plot((15, 15), (0, 11), 'k-', linewidth=27, label='top wall')
    # plt.plot((15, 15), (19, 30), 'k-', linewidth=27, label='bottom wall')
    # circle = plt.Circle((5, 5), radius=0.12)
    # ax = plt.gca()
    # ax.add_patch(circle)

    #plt.gca().set_autoscale(False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_xlim(0, p.x_dim)
    plt.gca().set_ylim(0, p.y_dim)
    plt.xlabel('X - Coordinates')
    plt.ylabel('Y - Coordinates')
    plt.savefig('RobotPath.png')
    # plt.show()


def create_plots(srun, walls):
    create_learning_curve()
    create_robot_path_plot(srun, walls)

