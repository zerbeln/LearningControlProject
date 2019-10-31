import neural_network as neu_net
import evolutionary_algorithm as ev_alg
from parameters import Parameters as p

def main():
    nn = neu_net.NeuralNetwork(p)
    ea = ev_alg.EvoAlg(p)

main()
