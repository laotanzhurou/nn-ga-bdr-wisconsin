# import libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import GA_FeatureSelection as ga


# define a function to plot confusion matrix
def plot_confusion(input_sample, num_classes, des_output, actual_output):
    confusion = torch.zeros(num_classes, num_classes)
    for i in range(input_sample):
        actual_class = actual_output[i]
        predicted_class = des_output[i]

        confusion[actual_class][predicted_class] += 1

    return confusion

"""
Step 1: Load data and pre-process data
Here we use data loader to read data
"""

# Neural Network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out


def bimodal_remove(X, Y, output, sigma, variance_threshold):
    p_error = abs(Y.numpy() - output.detach().numpy())
    pe_mean = p_error.mean()
    pe_std = p_error.std()

    if pe_std < variance_threshold:
        return X, Y

    # plot error distribution
    # n, bins, patches = plt.hist(p_error, 10, range=[0, 2], facecolor='blue')
    # plt.xlabel('Error')
    # plt.ylabel('Frequency')
    # plt.title(r'Histogram of IQ: $\mu='+str(pe_mean)+'$, $\sigma='+str(pe_std)+'$')
    # plt.tight_layout()
    # plt.show()

    # create candidate array
    candidate = []
    for i in range(0, len(p_error)):
        if p_error[i] > pe_mean:
            candidate.append(p_error[i])
    candidate = np.array(candidate)

    # create removal array
    c_mean = candidate.mean()
    c_std = candidate.std()
    removal = []
    for j in range(0, len(candidate)):
        if candidate[j] > c_mean + sigma * c_std:
            removal.append(candidate[j])
    removal = np.array(removal)

    # prepare index to remove
    removal_index = []
    for k in range(0, len(p_error)):
        for l in range(0, len(removal)):
            if p_error[k] == removal[l]:
                removal_index.append(k)

    # remove from training set
    x_num = X.numpy()
    y_num = Y.numpy()
    x_num = np.delete(x_num, removal_index, 0)
    y_num = np.delete(y_num, removal_index, 0)
    updated_x = torch.from_numpy(x_num)
    updated_y = torch.from_numpy(y_num)
    return updated_x, updated_y


def training(X, Y, bdr, sigma, variance_threshold, input_size, hidden_size, num_classes):
    net = Net(input_size, hidden_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # store all losses for visualisation
    all_losses = []

    # train the model
    end_epoch = 0
    for epoch in range(num_epochs):
        end_epoch = epoch
        # Forward
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(X)

        # Compute loss
        loss = criterion(outputs, Y)
        all_losses.append(loss.item())

        # Bimodal Removal
        if (epoch % 50 == 0):
            _, predicted = torch.max(outputs, 1)
            # calculate and print accuracy
            # total = predicted.size(0)
            # correct = predicted.data.numpy() == Y.data.numpy()
            # print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
            #       % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct) / total))

            # perform bimodal removal
            if bdr:
                X, Y = bimodal_remove(X, Y, _, sigma, variance_threshold)
                # print("Training Set Size after Bimodal Removal: " + str(Y.size()[0]))

        # Backward
        net.zero_grad()
        loss.backward()
        optimizer.step()

    # Plot loss
    # plt.figure()
    # plt.plot(all_losses)
    # plt.show()
    return net, end_epoch + 1, Y.size()[0]


def evaluate(net, test_data, updated_input_size):
    # get testing data
    test_input = test_data[:, np.arange(updated_input_size)]
    test_target = test_data[:, updated_input_size]

    inputs = torch.from_numpy(test_input).float()
    targets = torch.from_numpy(test_target).long()

    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)

    total = predicted.size(0)
    correct = predicted.data.numpy() == targets.data.numpy()

    print('Testing Accuracy: %.2f %%' % (100 * sum(correct) / total))
    return sum(correct) / total


def feature_selection(feature_input, label_input):
    return ga.feture_reduction(feature_input, label_input)


# Hyper Parameters
input_size = 30
hidden_size = 40
num_classes = 2
num_epochs = 500
learning_rate = 0.02
test_variance_threshold = 0.01
rounds = 10
use_ga = True
use_bdr = True

all_accuracy = []
all_bdr_size = []
all_ga_size = []
all_ending_epoch = []

# sigma_series = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
sigma_series = [1]

for test_sigma in sigma_series:
    for a in range(0, rounds):
        # load all data
        train_data = genfromtxt('data/wdbc_training.csv', delimiter=',')
        test_data = genfromtxt('data/wdbc_testing.csv', delimiter=',')

        # pre-processing: feature selection with GA
        if use_ga:
            reduced_features = feature_selection(train_data[:, np.arange(input_size)], train_data[:, input_size])
            train_data = np.delete(train_data, reduced_features, axis=1)
            test_data = np.delete(test_data, reduced_features, axis=1)
            updated_input_size = train_data.shape[1]-1
            #print("feature after ga: " + str(updated_input_size))
        else:
            updated_input_size = input_size

        # define train dataset and a data loader
        train_input = train_data[:, np.arange(updated_input_size)]
        train_target = train_data[:, updated_input_size]

        X = torch.from_numpy(train_input).float()
        Y = torch.from_numpy(train_target).long()

        net, result_epoch, bdr_size = training(X, Y, use_bdr, test_sigma, test_variance_threshold, updated_input_size, hidden_size, num_classes)
        accuracy = evaluate(net, test_data, updated_input_size)

        all_accuracy.append(accuracy)
        all_bdr_size.append(bdr_size)
        all_ending_epoch.append(result_epoch)
        all_ga_size.append(updated_input_size)

    mean_accuracy = np.array(all_accuracy).mean()
    mean_bdr_size = np.array(all_bdr_size).mean()
    mean_end_epoch = np.array(all_ending_epoch).mean()
    mean_ga_size = np.array(all_ga_size).mean()

    print("Result from sigma: " + str(test_sigma) + " variance_threshold: " + str(test_variance_threshold)
          + " average accuracy: " + str(mean_accuracy)
          + " ending epoch: " + str(mean_end_epoch)
          + " features after GA: " + str(mean_ga_size)
          + " input set after BDR: " + str(mean_bdr_size))