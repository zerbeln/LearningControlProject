#!/usr/bin/env python

import neural_network as neu_net
import evolutionary_algorithm as ev_alg
from parameters import Parameters as p
from agent import Agent
from world import World


def try_it_out(nn, wld, ag):
    ag.reset_agent_to_start()
    reward = 0

    for step in range(100):
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

    # Test initial population
    for i in range(ea.total_pop_size):
        nn.get_nn_weights(ea.pops[i])
        reward = try_it_out(nn, wld, ag)
        print(ag.agent_pos, reward)
        ea.fitness[i] = reward

    ea.epsilon_greedy_select()
    ea.offspring_pop = ea.parent_pop.copy()  # Produce K offspring
    ea.mutate()  # Mutate offspring population

    # Train population
    # for gen in range(p.generations):
    #     for i in range(p.offspring_pop_size):
    #         nn.get_nn_weights(ea.pops[i])
    #         reward = try_it_out(nn, wld, ag)
    #         ea.offspring_fitness[i] = reward
    #         print(ag.agent_pos, reward)
    # ea.down_select()


main()
