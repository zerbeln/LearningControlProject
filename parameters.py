#!/usr/bin/env python

class Parameters:

    # Test Parameters
    stat_runs = 1  # Number of statistical runs to perform
    generations = 100  # Number of generations for training NN

    # Agent Parameters
    # Turtlebot3 Burger
    # http://emanual.robotis.com/docs/en/platform/turtlebot3/specifications/#hardware-specifications
    sensor_resolution = 360
    detection_radius = 3.5  # Meters
    agent_rad = 0.12  # Radius of turtlebot (used for size estimations and collision detection)
    max_vel = 0.22  # m/s
    max_rot_vel = 2.84  # rad / s
    min_dist_to_wall = 0.05  # If the agent gets closer than this, consider it a collision

    # World Parameters
    x_dim = 30.0
    y_dim = 30.0
    w_thick = 2.0  # Thickness of the walls in the world
    coll_penalty = -10.0
    stp_penalty = -1.0
    n_walls = 2  # Number of internal walls in the world
    d_length = 4.0  # Length of the door in the world

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

    # LIDAR parameters
    small_steps = 5  # For areas we care about, chunk up by 5 degree segments
    big_steps = 15  # For areas we don't care as much about, chunk up in 15 degree segments