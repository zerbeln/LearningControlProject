% Nick Zerbel
% ROB 537: Learning Based Control
% October 7, 2019

close all; clear all; clc

%% Parameters
eps = 100;
eps1 = 150;
eps2 = 200;
eps3 = 250;
eps4 = 300;

%% Input Data
% Squared Error
% Number of Hidden Nodes
four_hnodes_error = importdata('Output_Data/NetworkError_Four.csv');
six_hnodes_error = importdata('Output_Data/NetworkError_Six.csv');
eight_hnodes_error = importdata('Output_Data/NetworkError_Eight.csv');
ten_hnodes_error = importdata('Output_Data/NetworkError_Ten.csv');
twelve_hnodes_error = importdata('Output_Data/NetworkError_Twelve.csv');

% Training Steps
steps1_train_error = importdata('Output_Data/NetworkError_S1.csv');
steps2_train_error = importdata('Output_Data/NetworkError_S2.csv');
steps3_train_error = importdata('Output_Data/NetworkError_S3.csv');
steps4_train_error = importdata('Output_Data/NetworkError_S4.csv');

% Learning Rate
eta0_train_error = importdata('Output_Data/NetworkError_E0.csv');
eta2_train_error = importdata('Output_Data/NetworkError_E2.csv');

% Performance
% Number of Hidden Nodes
four_performance = importdata('Output_Data/NetworkPerformance_Four.csv');
six_performance = importdata('Output_Data/NetworkPerformance_Six.csv');
eight_performance = importdata('Output_Data/NetworkPerformance_Eight.csv');
ten_performance = importdata('Output_Data/NetworkPerformance_Ten.csv');
twelve_performance = importdata('Output_Data/NetworkPerformance_Twelve.csv');

% Training Steps
steps1_performance = importdata('Output_Data/NetworkPerformance_S1.csv');
steps2_performance = importdata('Output_Data/NetworkPerformance_S2.csv');
steps3_performance = importdata('Output_Data/NetworkPerformance_S3.csv');
steps4_performance = importdata('Output_Data/NetworkPerformance_S4.csv');

% Learning Rate
eta0_performance = importdata('Output_Data/NetworkPerformance_E0.csv');
eta2_performance = importdata('Output_Data/NetworkPerformance_E2.csv');

%% Data Analysis
% Squared Error
% Number of Hidden Nodes
four_error = mean(four_hnodes_error.data);
six_error = mean(six_hnodes_error.data);
eight_error = mean(eight_hnodes_error.data);
ten_error = mean(ten_hnodes_error.data);
twelve_error = mean(twelve_hnodes_error.data);

% Number of Training Steps
s0_error = mean(eight_hnodes_error.data);
s1_error = mean(steps1_train_error.data);
s2_error = mean(steps2_train_error.data);
s3_error = mean(steps3_train_error.data);
s4_error = mean(steps4_train_error.data);

% Learning Rate
e0_error = mean(eta0_train_error.data);
e1_error = mean(eight_hnodes_error.data);
e2_error = mean(eta2_train_error.data);

% Performance
% Number of Hidden Nodes
four_perf = mean(four_performance.data);
six_perf = mean(six_performance.data);
eight_perf = mean(eight_performance.data);
ten_perf = mean(ten_performance.data);
twelve_perf = mean(twelve_performance.data);

% Number of Training Steps

s0_perf = mean(eight_performance.data);
s1_perf = mean(steps1_performance.data);
s2_perf = mean(steps2_performance.data);
s3_perf = mean(steps3_performance.data);
s4_perf = mean(steps4_performance.data);

% Learning Rate
e0_perf = mean(eta0_performance.data);
e1_perf = mean(eight_performance.data);
e2_perf = mean(eta2_performance.data);

%% Graph Generator
% Number of Hidden Nodes
figure()
X = [1:eps];
plot(X, four_error, 'Linewidth', 1.5)
hold on
plot(X, six_error, 'Linewidth', 1.5)
plot(X, eight_error, 'Linewidth', 1.5)
plot(X, ten_error, 'Linewidth', 1.5)
plot(X, twelve_error, 'Linewidth', 1.5)
xlabel('Epoch')
ylabel('Squared Network Error')
legend('4 Nodes', '6 Nodes', '8 Nodes', '10 Nodes', '12 Nodes')

figure()
y_data = [four_perf(:), six_perf(:), eight_perf(:), ten_perf(:), twelve_perf(:)];
x_data = [0, 1, 2, 3];
bar(x_data, y_data, 'grouped')
xlabel('Test Set')
ylabel('Percentage of Correctly Sorted Sets')
ylim([0, 100])
legend('4 Nodes', '6 Nodes', '8 Nodes', '10 Nodes', '12 Nodes')

% Number of Training Steps
figure()
Y_data = [s0_error(eps), s1_error(eps1), s2_error(eps2), s3_error(eps3), s4_error(eps4)];
X_data = [100, 150, 200, 250, 300];
plot(X_data, Y_data, 'k', 'Linewidth', 1.5)
xlabel('Number of Training Epochs')
ylabel('Final Network Squared Error')

figure()
tsteps_data = [s0_perf(:), s1_perf(:), s2_perf(:), s3_perf(:), s4_perf(:)];
x_data = [0, 1, 2, 3];
bar(x_data, tsteps_data, 'grouped')
xlabel('Test Set')
ylabel('Percentage of Correctly Sorted Sets')
ylim([0, 100])
legend('100 Steps', '150 Steps', '200 Steps', '250 Steps', '300 Steps')

% Learning Rate
figure()
plot(X, e0_error, 'Linewidth', 1.5)
hold on
plot(X, e1_error, 'Linewidth', 1.5)
plot(X, e2_error, 'Linewidth', 1.5)
xlabel('Epoch')
ylabel('Squared Network Error')
legend('Eta = 0.01', 'Eta = 0.05', 'Eta = 0.1')

figure()
lrate_data = [e0_perf(:), e1_perf(:), e2_perf(:)];
x_data = [0, 1, 2, 3];
bar(x_data, lrate_data, 'grouped')
xlabel('Test Set')
ylabel('Percentage of Correctly Sorted Sets')
ylim([0, 100])
legend('Eta = 0.01', 'Eta = 0.05', 'Eta = 0.1')

figure()
y_data = [eight_perf(2:4)];
x_data = [1, 2, 3];
bar(x_data, y_data)
xlabel('Test Set')
ylabel('Percentage of Correctly Sorted Sets')