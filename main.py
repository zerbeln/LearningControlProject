#!/usr/bin/env python

import neural_network as neu_net
import evolutionary_algorithm as ev_alg
from parameters import Parameters as p
from agent import Agent
from world import World

def main():
    nn = neu_net.NeuralNetwork(p)
    nn.create_nn_weights()
    ea = ev_alg.EvoAlg(p)
    wld = World(p)
    ag = Agent(p)
    ag.set_agent_start_pos()
    wld.world_config1()
    reward = 0

    #go lidar go. initial sweep
    ea.reset_populations()
    #order of operations
    #sweep, downsize, pass to neural net, receive movement, move, update reward/collision
    #repeat until steps are done
    #ten seconds of movement
    for step in range(100):
        sweep = ag.lidar_scan(wld.world_x,wld.world_y,wld.walls)
        nn.downsample_lidar(sweep)
        nn.get_outputs()
        ag.agent_step(nn.out_layer,p.time_step)
        reward += wld.calculate_reward(ag.agent_pos)
        print(ag.agent_pos, reward)

main()
