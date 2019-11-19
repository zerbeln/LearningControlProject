#!/usr/bin/env python

import neural_network as neu_net
import evolutionary_algorithm as ev_alg
from parameters import Parameters as p
from agent import Agent
from world import World


def main():
    # initialize
    nn = neu_net.NeuralNetwork(p)
    ea = ev_alg.EvoAlg(p)
    wld = World(p)
    ag = Agent(p)
    ag.set_agent_start_pos()
    wld.world_config1()
    ea.reset_populations()

    # order of operations
    # sweep, downsize, pass to neural net, receive movement, move, update reward/collision
    # repeat until steps are done
    # ten seconds of movement

    # Test initial population
    for i in range(ea.total_pop_size):
        ag.reset_agent_to_start()
        nn.get_nn_weights(ea.pops[i])
        reward = 0
        goal = False
        collision = False
        for step in range(100):
            # goal = ag.goal_reached(wld.door)#what do we want to do when we have passed the threshold
            sweep = ag.lidar_scan(wld.world_x, wld.world_y, wld.walls)
            nn.downsample_lidar(sweep)
            nn.get_outputs()
            ag.agent_step(nn.out_layer, p.time_step)
            # stop if there is a collision
            collision = ag.collision_detection()

            # stop if we reached the goal and did not collide with something
            goal = ag.goal_reached(wld.door)
            if goal:
                reward += ag.calculate_reward(goal, collision)
                break
            # calculate reward
            reward += ag.calculate_reward(goal, collision)
        print(ag.agent_pos, reward)
        ea.fitness[i] = reward
    ea.epsilon_greedy_select()
    ea.offspring_pop = ea.parent_pop.copy()  # Produce K offspring
    ea.mutate()  # Mutate offspring population
    # Train population
    for gen in range(p.generations):
        for i in range(p.offspring_pop_size):
            ag.reset_agent_to_start()
            nn.get_nn_weights(ea.offspring_pop[i])
            reward = 0
            for step in range(100):
                goal = ag.goal_reached(wld.door)  # what do we want to do when we have passed the threshold
                sweep = ag.lidar_scan(wld.world_x, wld.world_y, wld.walls)
                nn.downsample_lidar(sweep)
                nn.get_outputs()
                ag.agent_step(nn.out_layer, p.time_step, p.max_vel, p.max_rot_vel)
                goal = ag.goal_reached(wld.door)
                collision = ag.collision_detection()
                reward += ag.calculate_reward(goal, collision)
                # print(ag.agent_pos, reward)
            ea.offspring_fitness[i] = reward
            print(ag.agent_pos, reward)
    ea.down_select()
main()
