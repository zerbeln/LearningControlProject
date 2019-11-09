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

    #go lidar go. initial sweep
    sweep = ag.lidar_scan(wld.world_x,wld.world_y,wld.walls)
    #order of operations
    #sweep, downsize, pass to neural net, receive movement, move
    #repeat until steps are done
    downLIDAR = nn.downsample_lidar(sweep)
    print(downLIDAR)
    # nn.get_inputs(downLIDAR)
    # nn.get_outputs()
    # print(output)
    # ag.agent_step(nn_outputs, time_step)

main()
