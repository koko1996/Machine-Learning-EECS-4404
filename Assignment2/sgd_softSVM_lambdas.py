import csv
import numpy as np
from random import randint
import matplotlib.pyplot as plt

# hingeLoss returns the average hinge loss on the data given the weight vector 
# @param input_data is a matrix where the first column is the class (1 or -1)
# and columns [1:n] is the feature vector
# @param weight is the weight vector for a hyperplane
def hingeLoss( input_data, weight ):
    loss = 0.0
    for row in input_data:
        loss = loss + float(max(0 , 1 - row[0] * weight.dot(row[1:])))
    return loss/len(input_data)

# binaryLoss returns the average binary loss on the data given the weight vector 
# @param input_data is a matrix where the first column is the class (1 or -1)
# and columns [1:n] is the feature vector
# @param weight is the weight vector for a hyperplane
def binaryLoss( input_data, weight ):
    loss = 0.0
    for row in input_data:
        if row[0] * weight.dot(row[1:]) <= 0 :
            loss = loss + 1.0
    return loss/len(input_data)


# Main 
data = np.genfromtxt(open("bg.txt", "rb"), delimiter=",", dtype="float")
N = (len(data))
ones = np.ones((N,1), dtype=float)          # create a column of ones for the bias
data = np.append(ones, data, axis=1)        # add the ones column to data

ones = np.ones((N/2,1), dtype=float)        # create a column of ones for the class type
neg_ones = np.negative(ones)                # create a column of negative ones 
add_column = np.concatenate((ones, neg_ones))   
data = np.append(add_column, data, axis=1)  # add the column that represents the class types

for lambd in [100.0,10.0,1.0,.1,.01,.001]:
    hinge = []
    binary = []
    alpha = np.zeros((1,np.shape(data)[1] - 1 ), dtype=float)       # initialize alpha to zero 
    for j in range(1,500):                                          # iterate 
        w = (1.0/ (j * lambd) ) * alpha                             # update the scale of weight vector
        index = randint(0, len(data)-1)                             # pick the random index
        if (data[index][0] * w.dot(data[index][1:])) < 1:           # check if it gets classified correctly
            alpha = alpha + (data[index][0] * data[index][1:])      # add the misclassified vector 
        hinge.append(hingeLoss(data,w))                             # append hinge loss
        binary.append(binaryLoss(data,w))                           # append binary loss

    plt.plot(list(range(1,len(hinge)+1)), hinge, color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('hinge loss with lambda ' + str(lambd))
    plt.show()

    plt.plot(list(range(1,len(binary)+1)), binary, color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('binary loss with lambda ' + str(lambd))
    plt.show()