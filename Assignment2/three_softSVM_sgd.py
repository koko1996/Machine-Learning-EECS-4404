import csv
import numpy as np
from random import randint
import matplotlib.pyplot as plt

# binaryLoss returns the average binary loss on the data given the weight vector 
# @param input_data is a matrix where the last column is the class (1,2 or 3)
# and columns [0:n-1] is the feature vector
# @param weight is the weight vector for a hyperplane
# @param class_type integer value (1,2 or 3) that assumes the input vector class 
# type is 1 if it matches the given class_type otherwise it assumes the input vector
# class type is -1
def binaryLoss( input_data, weight, class_type ):
    loss = 0.0
    for row in input_data:
        t = row[-1]
        if row[-1] != class_type:
            t = -1
        if t * weight.dot(row[:-1]) <= 0 :
            loss = loss + 1.0
    return loss/len(input_data)

# softSVM returns the array of binary loss for each weight vector calculated during the
# stochastic gradient descent loop
# @param input_data is a matrix where the last column is the class (1,2 or 3)
# and columns [0:n-1] is the feature vector
# @param weight is the weight vector for a hyperplane
# @param class_type integer value (1,2 or 3) that assumes the input vector class 
# type is 1 if it matches the given class_type otherwise it assumes the input vector
# class type is -1
def softSVM (input_data, class_type):
    binary = []
    alpha = np.zeros((1,np.shape(input_data)[1] - 1 ), dtype=float)
    for j in range(1,500):
        w = (1.0/j) * alpha
        index = randint(0, len(input_data)-1)
        t = input_data[index][-1]
        if input_data[index][-1] != class_type:
            t = -1
        if (t * w.dot(input_data[index][:-1])) < 1:
            alpha = alpha + (t * input_data[index][:-1])
        binary.append(binaryLoss(input_data,w,class_type))
    return binary
        

# Main
with open('seeds_dataset.txt') as f:
    reader = csv.reader(f, delimiter='\t')
    rows = list(reader)
data = []
for element in rows:
    next_row =  [i for i in element if i != '']
    data.append(next_row)
     
data=np.array([np.array(xi,dtype=float) for xi in data])
N = len(data)
ones = np.ones((N,1), dtype=float)
data = np.append(ones, data, axis=1)

binary_1 = softSVM(data, 1)
plt.plot(list(range(1,len(binary_1)+1)), binary_1, color='blue')
plt.xlabel('Iteration')
plt.ylabel('binary_1 loss')
plt.savefig('binary_loss_weight_1' + '.png')
plt.show()

binary_2 = softSVM(data, 2)
plt.plot(list(range(1,len(binary_2)+1)), binary_2, color='blue')
plt.xlabel('Iteration')
plt.ylabel('binary_2 loss')
plt.savefig('binary_loss_weight_2' + '.png')
plt.show()

binary_3 = softSVM(data, 3)
plt.plot(list(range(1,len(binary_3)+1)), binary_3, color='blue')
plt.xlabel('Iteration')
plt.ylabel('binary_3 loss')
plt.savefig('binary_loss_weight_3' + '.png')
plt.show()