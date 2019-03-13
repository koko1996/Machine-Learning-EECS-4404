import csv
import numpy as np
from random import randint
from pprint import pprint
import matplotlib.pyplot as plt


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

def binaryLoss( input_data, weight_1, weight_2, weight_3):
    loss = 0.0
    for row in input_data:
        if row[-1] != multiClassPrediction(row[:-1],weight_1, weight_2, weight_3):
            loss = loss + 1.0
    return loss/len(input_data)

def binaryLossClassType( input_data, weight, class_type ):
    loss = 0.0
    for row in input_data:
        t = row[-1]
        if row[-1] != class_type:
            t = -1
        if t * weight.dot(row[:-1]) <= 0 :
            loss = loss + 1.0
    return loss/len(input_data)

def softSVM (input_data, class_type):
    w_best= np.zeros((1,np.shape(input_data)[1] - 1 ), dtype=float)    
    min_loss= float("inf")
    alpha = np.zeros((np.shape(input_data)[1] - 1 ), dtype=float)
    for j in range(1,100000):
        w = (0.01) * alpha
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
for element in rows:
    next_row =  [i for i in element if i != '']
    data.append(next_row)
     
np.set_printoptions(threshold=np.inf)
     
data=np.array([np.array(xi,dtype=float) for xi in data])
N = len(data)
ones = np.ones((N,1), dtype=float)
standardized_data =  (data[:,:-1] - np.mean(data[:,:-1],axis=0))  / np.std(data[:,:-1],axis=0)
final_data = np.append(standardized_data, data[:,-1].reshape(N, 1), axis=1)
final_data = np.append(ones, final_data, axis=1)


w_1 = softSVM(final_data, 1)
w_2 = softSVM(final_data, 2)
w_3 = softSVM(final_data, 3)

print("\n")
print (binaryLoss(final_data,w_1,w_2,w_3))