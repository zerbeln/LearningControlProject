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

main()
