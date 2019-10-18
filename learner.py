from parameters import Parameters as p
import numpy as np

def calculate_deltas(output_error, nn):
    output_deltas = np.zeros(p.num_outputs)
    hidden_deltas = np.zeros(p.num_hidden)

    # Calculate output deltas
    output_deltas[0] = output_error[0] * nn.out_layer[0] * (1 - nn.out_layer[0])
    output_deltas[1] = output_error[1] * nn.out_layer[1] * (1 - nn.out_layer[1])

    # Calculate hidden deltas
    weight_count = 0
    for j in range(p.num_hidden):
        for k in range(p.num_outputs):
            hidden_deltas[j] += nn.weights2[weight_count]*output_deltas[k]*nn.hid_layer[j]*(1-nn.hid_layer[j])
            weight_count += 1

    return output_deltas, hidden_deltas

def update_nn_weights(output_deltas, hidden_deltas, nn):

    # Update input-hidden layer weights
    weight_count = 0
    for i in range(p.num_inputs):
        for j in range(p.num_hidden):
            dw = p.eta * hidden_deltas[j]*nn.in_layer[i]
            nn.weights1[weight_count] += dw
            weight_count += 1

    # Update hidden-output layer weights
    weight_count = 0
    for j in range(p.num_hidden):
        for k in range(p.num_outputs):
            dw = p.eta * output_deltas[k]*nn.hid_layer[j]
            nn.weights2[weight_count] += dw
            weight_count += 1
