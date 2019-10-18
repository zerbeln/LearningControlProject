import neural_network as neu_net
import numpy as np
import csv
import os
from parameters import Parameters as p
from learner import calculate_deltas, update_nn_weights

def get_data(filename):  # Read data in from provided CSVs
    input_array = np.zeros((2, 100))
    output_array = np.zeros((2, 100))

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            for i in range(2):
                input_array[0, line_count] = float(row[0])
                input_array[1, line_count] = float(row[1])
            # print(input_array[:, line_count])
            for o in range(2):
                output_array[0, line_count] = float(row[2])
                output_array[1, line_count] = float(row[3])
            # print(output_array[:, line_count])
            line_count += 1

    return input_array, output_array

def create_output_files(file_name, in_vec):  # Create output CSVs for graphs
    dir_name = 'Output_Data/'

    if not os.path.exists(dir_name):  # If directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, file_name)
    with open(save_file_name, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Error'] + in_vec)

def main():
    nn = neu_net.NeuralNetwork()

    for srun in range(p.stat_runs):
        print('STAT RUN: ', srun)
        nn_performance = [0, 0, 0, 0]  # Tracks the number of correctly sorted sets for each test data set
        nn.create_nn_weights()
        error = np.zeros((2, p.n_sets))
        x_array, y_array = get_data(p.train1)
        nn_error = []
        for ep in range(p.n_episodes):
            nn_classification = np.ones((2, p.n_sets)) * -1
            for n in range(p.n_sets):
                nn.get_inputs(x_array[:, n])
                nn.get_outputs()

                # Classify data set
                if nn.out_layer[0] < 0.5 and nn.out_layer[1] > 0.5:  # [0, 1] Pass
                    nn_classification[0, n] = 0
                    nn_classification[1, n] = 1
                elif nn.out_layer[0] > 0.5 and nn.out_layer[1] < 0.5:  # [1, 0] Fail
                    nn_classification[0, n] = 1
                    nn_classification[1, n] = 0
                else:  # Classification error
                    nn_classification[0, n] = -1
                    nn_classification[1, n] = -1

                # Calculate network error
                error[0, n] = y_array[0, n] - nn.out_layer[0]
                error[1, n] = y_array[1, n] - nn.out_layer[1]

                # Calculate Delta
                out_deltas, hid_deltas = calculate_deltas(error[:, n], nn)

                # Update Weights
                update_nn_weights(out_deltas, hid_deltas, nn)

            # Calculate Mean Square Error
            squared_error = 0.0
            for n in range(p.n_sets):
                squared_error += (error[0, n]**2)
                squared_error += (error[1, n]**2)
            squared_error /= 2
            nn_error.append(squared_error)

            # Calculate correct classifications
            if ep == (p.n_episodes-1):
                correct_sorts = 0
                for n in range(p.n_sets):
                    if y_array[0, n] == nn_classification[0, n] and y_array[1, n] == nn_classification[1, n]:
                        correct_sorts += 1
                nn_performance[0] = correct_sorts

        create_output_files('NetworkError_E2.csv', nn_error)

        # Test trained NN
        if p.run_tests:

            # TEST 1 -------------------------------------------------------------------------------------------------
            x_array, y_array = get_data(p.test1)
            nn_classification = np.ones((2, p.n_sets)) * -1
            for n in range(p.n_sets):
                nn.get_inputs(x_array[:, n])
                nn.get_outputs()

                # Classify data set
                if nn.out_layer[0] < 0.5 and nn.out_layer[1] >= 0.5:  # [0, 1] Pass
                    nn_classification[0, n] = 0
                    nn_classification[1, n] = 1
                elif nn.out_layer[0] >= 0.5 and nn.out_layer[1] < 0.5:  # [1, 0] Fail
                    nn_classification[0, n] = 1
                    nn_classification[1, n] = 0
                else:  # Classification error
                    nn_classification[0, n] = -1
                    nn_classification[1, n] = -1

            # Calculate correct classifications
            correct_sorts = 0
            for n in range(p.n_sets):
                if y_array[0, n] == nn_classification[0, n] and y_array[1, n] == nn_classification[1, n]:
                    correct_sorts += 1
            nn_performance[1] = correct_sorts

            # TEST 2 --------------------------------------------------------------------------------------------------
            x_array, y_array = get_data(p.test2)
            nn_classification = np.ones((2, p.n_sets)) * -1
            for n in range(p.n_sets):
                nn.get_inputs(x_array[:, n])
                nn.get_outputs()

                # Classify data set
                if nn.out_layer[0] < 0.5 and nn.out_layer[1] >= 0.5:  # [0, 1] Pass
                    nn_classification[0, n] = 0
                    nn_classification[1, n] = 1
                elif nn.out_layer[0] >= 0.5 and nn.out_layer[1] < 0.5:  # [1, 0] Fail
                    nn_classification[0, n] = 1
                    nn_classification[1, n] = 0
                else:  # Classification error
                    nn_classification[0, n] = -1
                    nn_classification[1, n] = -1

            # Calculate correct classifications
            correct_sorts = 0
            for n in range(p.n_sets):
                if y_array[0, n] == nn_classification[0, n] and y_array[1, n] == nn_classification[1, n]:
                    correct_sorts += 1
            nn_performance[2] = correct_sorts

            # TEST 3 -------------------------------------------------------------------------------------------------
            x_array, y_array = get_data(p.test3)
            nn_classification = np.ones((2, p.n_sets)) * -1
            for n in range(p.n_sets):
                nn.get_inputs(x_array[:, n])
                nn.get_outputs()

                # Classify data set
                if nn.out_layer[0] < 0.5 and nn.out_layer[1] >= 0.5:  # [0, 1] Pass
                    nn_classification[0, n] = 0
                    nn_classification[1, n] = 1
                elif nn.out_layer[0] >= 0.5 and nn.out_layer[1] < 0.5:  # [1, 0] Fail
                    nn_classification[0, n] = 1
                    nn_classification[1, n] = 0
                else:  # Classification error
                    nn_classification[0, n] = -1
                    nn_classification[1, n] = -1

            # Calculate correct classifications
            correct_sorts = 0
            for n in range(p.n_sets):
                if y_array[0, n] == nn_classification[0, n] and y_array[1, n] == nn_classification[1, n]:
                    correct_sorts += 1
            nn_performance[3] = correct_sorts

            create_output_files('NetworkPerformance_E2.csv', nn_performance)

main()
