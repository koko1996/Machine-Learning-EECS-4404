import csv
import numpy as np
from random import randint
from pprint import pprint
import matplotlib.pyplot as plt


# multiClassPrediction returns the class of input vector based on one versus all method
# @param feature_vector is a vector where the last column is the class (1,2 or 3)
# and columns [0:n-1] is the feature input vector
# @param weight_1 is the weight vector for seperating class 1 from classes 2 and 3
# @param weight_2 is the weight vector for seperating class 2 from classes 1 and 3
# @param weight_3 is the weight vector for seperating class 3 from classes 1 and 2
def multiClassPrediction (feature_vector,  weight_1, weight_2, weight_3):
    result=0
    y_1 = weight_1.dot(feature_vector)
    y_2 = weight_2.dot(feature_vector)
    y_3 = weight_3.dot(feature_vector)
    if y_1 >= y_2 and y_1 >= y_3:
        result = 1
    elif y_2 >= y_3 and y_2 >= y_1:
        result = 2
    else :
        result = 3
    return result


# binaryLoss returns the average binary loss on the data based on one versus all method
# @param input_data is a matrix where the last column is the class (1,2 or 3)
# and columns [0:n-1] is the feature vector
# @param weight_1 is the weight vector for seperating class 1 from classes 2 and 3
# @param weight_2 is the weight vector for seperating class 2 from classes 1 and 3
# @param weight_3 is the weight vector for seperating class 3 from classes 1 and 2
def binaryLoss( input_data, weight_1, weight_2, weight_3):
    loss = 0.0
    for row in input_data:
        if row[-1] != multiClassPrediction(row[:-1],weight_1, weight_2, weight_3):
            loss = loss + 1.0
    return loss/len(input_data)


# binaryLossClassType returns the average binary loss on the data given the weight vector 
# and class type
# @param input_data is a matrix where the last column is the class (1,2 or 3)
# and columns [0:n-1] is the feature vector
# @param weight is the weight vector for a hyperplane
# @param class_type integer value (1,2 or 3) that assumes the input vector class 
# type is 1 if it matches the given class_type otherwise it assumes the input vector
# class type is -1
def binaryLossClassType( input_data, weight, class_type ):
    loss = 0.0
    for row in input_data:
        t = row[-1]
        if row[-1] != class_type:
            t = -1
        if t * weight.dot(row[:-1]) <= 0 :
            loss = loss + 1.0
    return loss/len(input_data)



# softSVM returns weight vector for the maximum margin seperating hyperplane for 
# a binary classifier
# @param input_data is a matrix where the last column is the class (1,2 or 3)
# and columns [0:n-1] is the feature vector
# @param class_type integer value (1,2 or 3) that assumes the input vector class 
# type is 1 if it matches the given class_type otherwise it assumes the input vector
# class type is -1
# @param iteration is the number of iterations to run the svm for
def softSVM (input_data, class_type, iteration):
    w_best= np.zeros((1,np.shape(input_data)[1] - 1 ), dtype=float)    
    min_loss= float("inf")
    alpha = np.zeros((np.shape(input_data)[1] - 1 ), dtype=float)
    for j in range(1,iteration):
        w = (400000/j) * alpha
        index = randint(0, len(input_data)-1)
        t = input_data[index][-1]
        if input_data[index][-1] != class_type:
            t = -1
        if (t * w.dot(input_data[index][:-1])) < 1:
            alpha = alpha + (t * input_data[index][:-1])
        curr_loss = binaryLossClassType(input_data,w,class_type)
        if  curr_loss < min_loss:
            w_best = w
            min_loss = curr_loss
    return w_best



# main
with open('seeds_dataset.txt') as f:
    reader = csv.reader(f, delimiter='\t')
    rows = list(reader)

data = []

# preprocess the data by removing empty elements and      
for element in rows:
    next_row =  [i for i in element if i != '']
    data.append(next_row)
     
np.set_printoptions(threshold=np.inf)
data=np.array([np.array(xi,dtype=float) for xi in data])        
N = len(data)
ones = np.ones((N,1), dtype=float)
# standardize the dataset to decrease the loss
standardized_data =  (data[:,:-1] - np.mean(data[:,:-1],axis=0))  / np.std(data[:,:-1],axis=0)
final_data = np.append(standardized_data, data[:,-1].reshape(N, 1), axis=1)
final_data = np.append(ones, final_data, axis=1)


for iteration in range(1,1000000):
    w_1 = softSVM(final_data, 1, iteration)
    w_2 = softSVM(final_data, 2, iteration)
    w_3 = softSVM(final_data, 3, iteration)

    print("\n")
    print(w_1)
    print(w_2)
    print(w_3)
    print ("loss: " + str(binaryLoss(final_data,w_1,w_2,w_3)))