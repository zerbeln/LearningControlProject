#!/usr/bin/env python

import numpy as np


class NeuralNetwork:

    def __init__(self, p):
        self.n_inputs = p.num_inputs  # Number of inputs from laser scan
        self.n_outputs = p.num_outputs  # Number of TWIST message required by robot
        self.n_nodes = p.num_hidden  # Number of nodes in hidden layer
        self.n_hidden_weights = (self.n_inputs + 1)*self.n_nodes  # +1 is to include biasing node
        self.n_output_weights = (self.n_nodes + 1)*self.n_outputs  # +1 is to include biasing node
        self.n_weights = (self.n_inputs + 1)*self.n_nodes + (self.n_nodes + 1)*self.n_outputs  # Total number of weights
        self.input_bias = 1.0
        self.hidden_bias = 1.0
        self.weights1 = np.zeros(self.n_hidden_weights)
        self.weights2 = np.zeros(self.n_output_weights)
        self.in_layer = np.zeros(self.n_inputs+1)
        self.hid_layer = np.zeros(self.n_nodes)
        self.out_layer = np.zeros(self.n_outputs)
        self.sm_step = p.small_steps
        self.bg_step = p.big_steps

    def reset_nn(self):  # Clear current network
        """
        Clears neural network arrays so that they all contain zeros
        :return: None
        """
        self.weights1 = np.zeros(self.n_hidden_weights)
        self.weights2 = np.zeros(self.n_output_weights)
        self.in_layer = np.zeros(self.n_inputs+1)
        self.hid_layer = np.zeros(self.n_nodes)
        self.out_layer = np.zeros(self.n_outputs)

    def create_nn_weights(self):
        for w in range(self.n_hidden_weights):
            weight = np.random.normal(0, 1)
            self.weights1[w] = weight

        for w in range(self.n_output_weights):
            weight = np.random.normal(0, 1)
            self.weights2[w] = weight

    def get_inputs(self, input_vec):  # Get inputs from state-vector
        """
        Fill in input layer of neural network with sensor inputs from robot
        :param input_vec: Inputs from LIDAR sensors
        :return: None
        """

        for i in range(self.n_inputs + 1):
            if i < self.n_inputs:
                self.in_layer[i] = input_vec[i]
            else:
                self.in_layer[i] = self.input_bias

    def reset_layers(self):  # Clear hidden layers and output layers
        """
        Clear hidden layer and output layer before matrix operations
        :return: None
        """
        self.hid_layer = np.zeros(self.n_nodes)
        self.out_layer = np.zeros(self.n_outputs)

    def get_outputs(self):
        """
        Run NN to receive rover action outputs
        :param rov_id:
        :return: None
        """
        self.reset_layers()

        # Reshape weight arrays into a matrix for matrix multiplication
        ih_weights = np.reshape(self.weights1, [self.n_inputs + 1, self.n_nodes])
        ho_weights = np.reshape(self.weights2, [self.n_nodes + 1, self.n_outputs])

        self.hid_layer = np.dot(self.in_layer, ih_weights)
        self.hid_layer = np.append(self.hid_layer, self.hidden_bias)  # Append biasing node to hidden layer


        for n in range(self.n_nodes):  # Pass hidden layer nodes through activation function
            self.hid_layer[n] = self.sigmoid(self.hid_layer[n])

        self.out_layer = np.dot(self.hid_layer, ho_weights)

        for n in range(self.n_outputs):  # Pass output nodes through activation function
            self.out_layer[n] = self.sigmoid(self.out_layer[n])


    def tanh(self, inp):  # Tanh function as activation function
        """
        NN activation function
        :param inp: Node value in NN (pre-activation function)
        :return: tanh value
        """
        tanh = (2/(1 + np.exp(-2*inp)))-1
        return tanh

    def sigmoid(self, inp):  # Sigmoid function as activation function
        """
        NN activation function
        :param inp: Node value in NN (pre-activation function)
        :return: sigmoid value
        """
        sig = 1/(1 + np.exp(-inp))
        return sig

    def downsample_lidar(self, lidar):
        """
        Takes in LIDAR and downsamples with the following parameters:
            For the 90 degrees directly in front and behind (e.g. +/- 45 degrees from dead ahead), downsamples into
            chunks defined by self.sm_step (defined in parameters file)
            For the areas to the sides of the robot, it downsamples into chunks defined by self.bg_step (again, defined
            in parameters file)
        Then sets the downsampled LIDAR scan as the inputs to the NN with a 1 appended to the end for the bias
        :param lidar:
        :return:
        """

        stop0 = 0
        stop1 = 45
        stop2 = 135
        stop3 = 225
        stop4 = 315
        stop5 = 360

        new_scan = []
        i = 0

        while i < stop5:
            if stop0 <= i < stop1 or stop2 <= i < stop3 or stop4 <= i < stop5:
                new_scan.append(np.amin(lidar[i:i + self.sm_step]))
                i += self.sm_step
            else:
                new_scan.append(np.amin(lidar[i:i + self.bg_step]))
                i += self.bg_step

        new_scan.append(1)  # For bias neuron
        self.in_layer = np.array(new_scan)
        #return back array for now cuz I need to follow wtf is going on
        return new_scan
