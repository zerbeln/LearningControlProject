#!/usr/bin/env python

import neural_network as neu_net
import evolutionary_algorithm as ev_alg
from parameters import Parameters as p
from agent import Agent
from world import World
import numpy as np
import os
import csv

def try_network(nn, wld, ag):
    ag.reset_agent_to_start()
    reward = 0

    # Agent moves for n steps
    for step in range(p.agent_steps):
        sweep = ag.lidar_scan(wld.world_x, wld.world_y, wld.walls)
        nn.get_outputs(sweep)
        collision = ag.agent_step(nn.out_layer, p.time_step, wld.walls, wld.world_x, wld.world_y)

        # calculate reward
        step_reward, goal = wld.calculate_reward(ag.agent_pos, ag.body_radius, collision)
        reward += step_reward

        # Stop if we reached the goal
        if goal:
            break

    return reward

def test_best_network(nn, wld, ag):
    ag.reset_agent_to_start()
    reward = 0

    robot_path = [[0.0, 0.0, 0.0] for _ in range(p.agent_steps + 1)]
    robot_path[0][0] = ag.agent_pos[0]
    robot_path[0][1] = ag.agent_pos[1]
    robot_path[0][2] = ag.agent_pos[2]

    # Agent moves for n steps
    for step in range(p.agent_steps):
        sweep = ag.lidar_scan(wld.world_x, wld.world_y, wld.walls)
        nn.get_outputs(sweep)
        collision = ag.agent_step(nn.out_layer, p.time_step, wld.walls, wld.world_x, wld.world_y)
        robot_path[step+1][0] = ag.agent_pos[0]
        robot_path[step+1][1] = ag.agent_pos[1]
        robot_path[step+1][2] = ag.agent_pos[2]

        # calculate reward
        step_reward, goal = wld.calculate_reward(ag.agent_pos, ag.body_radius, collision)
        reward += step_reward

        # Stop if we reached the goal
        if goal:
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


def main():
    # initialize
    nn = neu_net.NeuralNetwork(p)
    ea = ev_alg.EvoAlg(p)
    wld = World(p)
    ag = Agent(p)
    ag.set_agent_start_pos()
    wld.world_config1()
    wld.set_agent_starting_room(ag.agent_pos)
    print("Starting Room is: ", wld.agent_starting_room)
    ea.reset_populations()

    # order of operations
    # sweep, downsize, pass to neural net, receive movement, move, update reward/collision
    # repeat until steps are done
    # ten seconds of movement

    best_fit = [0 for _ in range(p.generations+1)]

    # Test initial population
    for i in range(ea.total_pop_size):
        nn.get_nn_weights(ea.pops[i])
        reward = try_network(nn, wld, ag)
        # print(ag.agent_pos, reward)
        ea.fitness[i] = reward

    best_fit[0] = max(ea.fitness)  # Records best initial fitness
    ea.epsilon_greedy_select()
    ea.offspring_pop = ea.parent_pop.copy()  # Produce K offspring
    ea.mutate()  # Mutate offspring population


    # Train population
    for gen in range(p.generations):
        print("Generation: ", gen)
        for i in range(p.offspring_pop_size):
            nn.get_nn_weights(ea.pops[i])
            reward = try_network(nn, wld, ag)
            ea.offspring_fitness[i] = reward
            # print(ag.agent_pos, reward)
        if gen < p.generations-1:  # Do not do down-select at the end of the final generation
            ea.down_select()
        if gen == p.generations-1:
            ea.combine_pops()
        best_fit[gen+1] = max(ea.fitness)  # Record best fitness after each gen
        print('Best Reward: ', max(ea.fitness))

    create_output_files("BestFit.csv", best_fit)  # Records best fitness for each gen for learning curve
    best_nn = np.argmax(ea.fitness)
    nn.get_nn_weights(ea.pops[best_nn])
    reward, robot_path = test_best_network(nn, wld, ag)
    print("The final reward is: ", reward)

    create_output_files("RobotPath.csv", robot_path)  # Records path taken by best neural network

    record_best_nn = [0.0 for _ in range(ea.policy_size)]
    for w in range(ea.policy_size):  # Need to convert to non np-array for csv writer
        record_best_nn[w] = ea.pops[best_nn, w]
    create_output_files("BestNN.csv", record_best_nn)  # Records best neural network found


main()
