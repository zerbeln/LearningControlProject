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
    plt.show()

def create_robot_path_plot(srun):
    x_axis = [0.0 for _ in range(p.agent_steps + 1)]
    y_axis = [0.0 for _ in range(p.agent_steps + 1)]
    data_input = [[] for _ in range(p.stat_runs)]

    with open('Output_Data/RobotPath.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == srun:
                for i in range(p.agent_steps + 1):
                    data_input[line_count].append(row[i])
            line_count += 1

    processed_data = []
    for l in data_input[0]:
        for sub_l in l.split(","):
            sub_l = sub_l.strip('][')
            processed_data.append(sub_l)

    for stp in range(p.agent_steps + 1):
        x_axis[stp] = processed_data[3*stp]
        y_axis[stp] = processed_data[3*stp + 1]

    plt.figure()
    plt.plot(x_axis, y_axis)
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.autoscale(False)
    plt.xlabel('X - Coordinates')
    plt.ylabel('Y - Coordinates')
    plt.show()

def show_plots():
    create_learning_curve()
    create_robot_path_plot(0)

show_plots()

