# import libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

# define a function to plot confusion matrix
def plot_confusion(input_sample, num_classes, des_output, actual_output):
    confusion = torch.zeros(num_classes, num_classes)
    for i in range(input_sample):
        actual_class = actual_output[i]
        predicted_class = des_output[i]

        confusion[actual_class][predicted_class] += 1

    return confusion


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


def training(hidden_size, learning_rate):
    """
    Step 1: Load Data

    """
    # load all data
    train_data = genfromtxt('data/wdbc_training.csv', delimiter=',')
    test_data = genfromtxt('data/wdbc_testing.csv', delimiter=',')

    # define train dataset and a data loader
    train_input = train_data[:, np.arange(input_size)]
    train_target = train_data[:, input_size]

    X = torch.from_numpy(train_input).float()
    Y = torch.from_numpy(train_target).long()

    net = Net(input_size, hidden_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # store all losses for visualisation
    all_losses = []

    """
    Step 2: Training

    """
    for epoch in range(num_epochs):

        # Forward
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(X)

        # Compute loss
        loss = criterion(outputs, Y)
        all_losses.append(loss.item())

        if (epoch % 50 == 0 or epoch == num_epochs - 1):
            output, predicted = torch.max(outputs, 1)
            # calculate and print accuracy
            total = predicted.size(0)
            correct = predicted.data.numpy() == Y.data.numpy()

            # print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
            #       % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct) / total))

            # plot pattern errors
            p_error = abs(Y.numpy() - output.detach().numpy())
            pe_mean = p_error.mean()
            pe_std = p_error.std()

            # plot error distribution
            # n, bins, patches = plt.hist(p_error, 10, range=[0, 2], facecolor='blue')
            # plt.xlabel('Error')
            # plt.ylabel('Frequency')
            # plt.title(r'Histogram of IQ: $\mu=' + str(pe_mean) + '$, $\sigma=' + str(pe_std) + '$')
            # plt.tight_layout()
            # plt.show()

        # Backward
        net.zero_grad()
        loss.backward()
        optimizer.step()

    # plt.figure()
    # plt.plot(all_losses)
    # plt.show()


    """
    Step 3: Test the neural network

    Pass testing data to the built neural network and get its performance
    """
    # get testing data
    test_input = test_data[:, np.arange(input_size)]
    test_target = test_data[:, input_size]

    inputs = torch.from_numpy(test_input).float()
    targets = torch.from_numpy(test_target).long()

    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)

    total = predicted.size(0)
    correct = predicted.data.numpy() == targets.data.numpy()

    print('Testing Accuracy: %.2f %%' % (100 * sum(correct) / total))

    """
    Evaluating the Results

    To see how well the network performs on different categories, we will
    create a confusion matrix, indicating for every glass (rows)
    which class the network guesses (columns).

    """

    # print('Confusion matrix for testing:')
    # print(plot_confusion(test_input.shape[0], num_classes, predicted.long().data, targets.data))

    # return accuracy rate
    return sum(correct) / total


# perform training
input_size = 30
hidden_unit = 40
num_classes = 2
learning_rate = 0.01
num_epochs = 500
rounds = 10

all_accuracy = []
for a in range(0, rounds):
    print("Round " + str(a) + " out of " + str(rounds))
    accuracy = training(hidden_unit, learning_rate)
    all_accuracy.append(accuracy)

mean_accuracy = np.array(all_accuracy).mean()
print("Result from hidden unit: " + str(hidden_unit) + " lr: " + str(learning_rate) + " average accuracy: " + str(mean_accuracy))
