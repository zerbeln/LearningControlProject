import numpy as np


class NeuralNetwork:

    def __init__(self):
        self.n_inputs = 2  # Number of inputs from laser scan
        self.n_outputs = 2  # Number of TWIST message required by robot
        self.n_nodes = 8  # Number of nodes in hidden layer
        self.n_hidden_weights = (self.n_inputs + 1)*self.n_nodes
        self.n_output_weights = (self.n_nodes + 1)*self.n_outputs
        self.n_weights = (self.n_inputs + 1)*self.n_nodes + (self.n_nodes + 1)*self.n_outputs
        self.hidden_bias = 1.0
        self.output_bias = 1.0
        self.weights1 = np.zeros(self.n_hidden_weights)
        self.weights2 = np.zeros(self.n_output_weights)
        self.in_layer = np.zeros(self.n_inputs+1)
        self.hid_layer = np.zeros(self.n_nodes)
        self.out_layer = np.zeros(self.n_outputs)

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
        Assign inputs from rover sensors to the input layer of the NN
        :param state_vec: Inputs from rover sensors
        :param rov_id: Current rover
        :return: None
        """
        for i in range(self.n_inputs + 1):
            if i < self.n_inputs:
                self.in_layer[i] = input_vec[i]
            else:
                self.in_layer[i] = self.hidden_bias

    def reset_layers(self):  # Clear hidden layers and output layers
        """
        Zeros hidden layer and output layer of NN
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


        ih_weights = np.reshape(self.weights1, [self.n_inputs + 1, self.n_nodes])
        ho_weights = np.reshape(self.weights2, [self.n_nodes + 1, self.n_outputs])

        self.hid_layer = np.dot(self.in_layer, ih_weights)
        self.hid_layer = np.append(self.hid_layer, self.output_bias)
        # print(self.hid_layer)


        for n in range(self.n_nodes):  # Pass hidden layer nodes through activation function
            self.hid_layer[n] = self.sigmoid(self.hid_layer[n])

        self.out_layer = np.dot(self.hid_layer, ho_weights)
        # print(self.out_layer)

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
