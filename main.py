#!/usr/bin/env python

import neural_network as neu_net
import evolutionary_algorithm as ev_alg
from parameters import Parameters as p
from graph_generator import create_plots
from agent import Agent
from world import World
import numpy as np
import os
import csv

def try_network(nn, wld, ag, wallDict):
    ag.reset_agent_to_start()
    reward = 0

    # Agent moves for n steps
    for step in range(p.agent_steps):
        sweep = ag.lidar_scan(wallDict)
        nn.get_outputs(sweep)
        collision = ag.agent_step(nn.out_layer, p.time_step, wld.walls, wld.world_x, wld.world_y, wld.threshold)

        # calculate reward
        step_reward, goal = wld.calculate_reward(ag.agent_pos, ag.body_radius, collision)
        reward += step_reward

        # Stop if we reached the goal
        if goal:
            break

    return reward

def test_best_network(nn, wld, ag, wallDict):
    ag.reset_agent_to_start()
    reward = 0

    # Record initial position of robot
    robot_path = [[0.0, 0.0, 0.0] for _ in range(p.agent_steps + 1)]
    robot_path[0][0] = ag.agent_pos[0]
    robot_path[0][1] = ag.agent_pos[1]
    robot_path[0][2] = ag.agent_pos[2]

    # Agent moves for n steps
    for step in range(p.agent_steps):
        sweep = ag.lidar_scan(wallDict)
        nn.get_outputs(sweep)
        collision = ag.agent_step(nn.out_layer, p.time_step, wld.walls, wld.world_x, wld.world_y, wld.threshold)

        # Record path of robot
        robot_path[step+1][0] = ag.agent_pos[0]
        robot_path[step+1][1] = ag.agent_pos[1]
        robot_path[step+1][2] = ag.agent_pos[2]

        # calculate reward
        step_reward, goal = wld.calculate_reward(ag.agent_pos, ag.body_radius, collision)
        reward += step_reward

        # Stop if we reached the goal
        if goal:
            while step < p.agent_steps:
                robot_path[step + 1][0] = ag.agent_pos[0]
                robot_path[step + 1][1] = ag.agent_pos[1]
                robot_path[step + 1][2] = ag.agent_pos[2]
                step += 1
            break

    return reward, robot_path

def create_output_files(filename, in_vec):
    dir_name = 'Output_Data/'

    if not os.path.exists(dir_name):  # If directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, filename)
    with open(save_file_name, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(in_vec)

def build_wall_dict(wld):

    wallDict = {}  # init wall/boundary dictionary
    # build boundary segments
    # format [X3,X4,Y3,Y4]
    wallDict["worldLeft"] = [0, 0, 0, wld.world_y]
    wallDict["worldRight"] = [wld.world_x, wld.world_x, 0, wld.world_y]
    wallDict["worldTop"] = [0, wld.world_x, wld.world_y, wld.world_y]
    wallDict["worldBottom"] = [0, wld.world_x, 0, 0]

    # build wall segments
    # format [X3,X4,Y3,Y4]
    wallDict["wall1"] = [wld.walls[0, 0, 0], wld.walls[0, 1, 0], wld.walls[0, 0, 1], wld.walls[0, 1, 1]]
    wallDict["wall2"] = [wld.walls[1, 0, 0], wld.walls[1, 1, 0], wld.walls[1, 0, 1], wld.walls[1, 1, 1]]

    return wallDict


def main():
    # initialize
    nn = neu_net.NeuralNetwork(p)
    ea = ev_alg.EvoAlg(p)

    # Create instances of agents and worlds
    wld1a = World(p); wld1a.world_config1()
    wld1b = World(p); wld1b.world_config1()

    wld2a = World(p); wld2a.world_config2()
    wld2b = World(p); wld2b.world_config2()

    wld3a = World(p); wld3a.world_config3()
    wld3b = World(p); wld3b.world_config3()

    wld4a = World(p); wld4a.world_config4()
    wld4b = World(p); wld4b.world_config4()

    # Initialize instances of agents in training worlds (Parameters, X, Y, Theta)
    agent_instances = []
    # Agent instances in World 1
    theta_1a = 261.0 * np.pi/180.0
    ag1a = Agent(p, 3.2, 5.0, theta_1a)  # wld1a
    agent_instances.append(ag1a)
    theta_1b = 43.0 * np.pi/180.0
    ag1b = Agent(p, 6.3, 6.2, theta_1b)  # wld1b
    agent_instances.append(ag1b)

    # Agent instances in World 2
    theta_2a = 300.0 * np.pi/180.0
    ag2a = Agent(p, 4.1, 2.5, theta_2a)  # wld2a
    agent_instances.append(ag2a)
    theta_2b = 145.0 * np.pi/180.0
    ag2b = Agent(p, 6.9, 6.5, theta_2b)  # wld2b
    agent_instances.append(ag2b)

    # Agent instances World 3
    theta_3a = 182.8 * np.pi / 180.0
    ag3a = Agent(p, 4.1, 2.2, theta_3a)
    agent_instances.append(ag3a)
    theta_3b = 323.3 * np.pi / 180.0
    ag3b = Agent(p, 6.1, 3.8, theta_3b)
    agent_instances.append(ag3b)

    # Agent instances World 4
    theta_4a = 21.0 * np.pi/180.0
    ag4a = Agent(p, 2.0, 3.5, theta_4a)
    agent_instances.append(ag4a)
    theta_4b = 102.2 * np.pi*180.0
    ag4b = Agent(p, 3.6, 5.9, theta_4b)
    agent_instances.append(ag4b)

    # Initialize instances of worlds for training set
    training_set = []
    # World Configuration 1
    wld1a.set_agent_starting_room(ag1a.agent_pos)
    wld1b.set_agent_starting_room(ag1b.agent_pos)
    training_set.append(wld1a)
    training_set.append(wld1b)

    # World Configuration 2
    wld2a.set_agent_starting_room(ag2a.agent_pos)
    wld2b.set_agent_starting_room(ag2b.agent_pos)
    training_set.append(wld2a)
    training_set.append(wld2b)

    # World Configuration 3
    wld3a.set_agent_starting_room(ag3a.agent_pos)
    wld3b.set_agent_starting_room(ag3b.agent_pos)
    training_set.append(wld3a)
    training_set.append(wld3b)

    # World Configuration 4
    wld4a.set_agent_starting_room(ag4a.agent_pos)
    wld4b.set_agent_starting_room(ag4b.agent_pos)
    training_set.append(wld4a)
    training_set.append(wld4b)

    # Build dictionaries for LIDAR scans
    wall_dicts = []
    wallDict1 = build_wall_dict(wld1a)
    wall_dicts.append(wallDict1)
    wall_dicts.append(wallDict1)
    wallDict2 = build_wall_dict(wld2a)
    wall_dicts.append(wallDict2)
    wall_dicts.append(wallDict2)
    wallDict3 = build_wall_dict(wld3a)
    wall_dicts.append(wallDict3)
    wall_dicts.append(wallDict3)
    wallDict4 = build_wall_dict(wld4a)
    wall_dicts.append(wallDict4)
    wall_dicts.append(wallDict4)

    # Create test instances (not training set)
    theta_test = 271.0 * np.pi/180.0
    ag_test = Agent(p, 5.7, 3.3, theta_test)
    wld_test = World(p)
    wld_test.world_config2()
    wld_test.set_agent_starting_room(ag_test.agent_pos)
    wallDictTest = build_wall_dict(wld_test)


    if not p.run_graph_only:
        for sr in range(p.stat_runs):
            print("Stat Run: ", sr)
            ea.reset_populations()
            best_fit = [0 for _ in range(p.generations+1)]

            # Evaluate initial population -----------------------------------------------------------------------------
            for i in range(ea.total_pop_size):
                nn.get_nn_weights(ea.pops[i])
                reward = 0
                for tw in range(p.n_train_worlds):
                    reward += try_network(nn, training_set[tw], agent_instances[tw], wall_dicts[tw])
                ea.fitness[i] = reward

            best_fit[0] = max(ea.fitness)  # Records best initial fitness
            ea.epsilon_greedy_select()
            ea.offspring_pop = ea.parent_pop.copy()  # Produce K offspring
            ea.mutate()  # Mutate offspring population

            # Train population ------------------------------------------------------------------------------------
            for gen in range(p.generations):
                print("Generation: ", gen)
                for i in range(p.offspring_pop_size):
                    nn.get_nn_weights(ea.pops[i])
                    reward = 0
                    for tw in range(p.n_train_worlds):
                        reward += try_network(nn, training_set[tw], agent_instances[tw], wall_dicts[tw])
                    ea.offspring_fitness[i] = reward
                if gen < p.generations-1:  # Do not do down-select at the end of the final generation
                    ea.down_select()
                if gen == p.generations-1:
                    ea.combine_pops()
                best_fit[gen+1] = max(ea.fitness)  # Record best fitness after each gen

            # Test best NN ------------------------------------------------------------------------------------------
            ag_test.reset_agent_to_start()
            best_nn = np.argmax(ea.fitness)
            nn.get_nn_weights(ea.pops[best_nn])
            reward, robot_path = test_best_network(nn, wld_test, ag_test, wallDictTest)

            # Create data files ------------------------------------------------------------------------------------
            create_output_files("BestFit.csv", best_fit)  # Records best fitness for each gen for learning curve
            create_output_files("RobotPath.csv", robot_path)  # Records path taken by best neural network
            record_best_nn = [0.0 for _ in range(ea.policy_size)]
            for w in range(ea.policy_size):  # Need to convert to non np-array for csv writer
                record_best_nn[w] = ea.pops[best_nn, w]
            create_output_files("BestNN.csv", record_best_nn)  # Records best neural network found

    create_plots(0, wld_test.walls)  # (stat-run, walls)


main()
