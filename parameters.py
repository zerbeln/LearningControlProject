
class Parameters:

    # Test Parameters
    stat_runs = 15
    n_episodes = 100  # Number of training episodes for NN
    n_sets = 100  # Number of data sets to be tested on
    train1 = 'data/train1.csv'
    test1 = 'data/test1.csv'
    test2 = 'data/test2.csv'
    test3 = 'data/test3.csv'
    run_tests = True  # False only trains without testing


    # Learning Parameters
    eta = 0.1  # Learning rate

    # Neural Network Parameters
    num_inputs = 2
    num_hidden = 8
    num_outputs = 2