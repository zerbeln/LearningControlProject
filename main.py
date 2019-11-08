#!/usr/bin/env python

import neural_network as neu_net
import evolutionary_algorithm as ev_alg
from parameters import Parameters as p
from agent import Agent
from world import World

def main():
    nn = neu_net.NeuralNetwork(p)
    ea = ev_alg.EvoAlg(p)
    wld = World(p)
    ag = Agent(p)
    ag.set_agent_start_pos()
    wld.world_config1()
    sweep = ag.lidar_scan(wld.world_x,wld.world_y,wld.walls)
    print(sweep)

main()
