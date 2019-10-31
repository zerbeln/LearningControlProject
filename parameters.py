
class Parameters:

    # Test Parameters
    stat_runs = 1  # Number of statistical runs to perform
    generations = 100  # Number of generations for training NN

    # EA Parameters
    epsilon = 0.1  # For e-greedy selection
    prob_mutation = 0.1  # The probability that a given weight will mutate
    mutation_rate = 0.01  # The maximum rate of change allowed from the mutation
    offspring_pop_size = 10
    parent_pop_size = 10

    # Neural Network Parameters
    num_inputs = 2  # Number of input nodes in the input layer
    num_hidden = 8  # Number of nodes in the hidden layer
    num_outputs = 2  # Number of output nodes in the output layer