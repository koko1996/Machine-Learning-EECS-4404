###################################################################################################
# EECS4404 Assignment 1                                                                           #
# Filename: RLM.py                                                                                #   
# Author: NANAH JI, KOKO                                                                          #
# Email: koko96@my.yorku.com                                                                      #
###################################################################################################


import csv
import numpy as np
from math import e
import matplotlib.pyplot as plt

# Read output data as floating point integers to the list
reader = csv.reader(open("dataset2_outputs.txt", "rt"))
output_data=[]
for row in reader: 
   output_data.append(float(row[0])) 

# convert the list to np array
y_train=np.array(output_data)

# Read input data as floating point integers to the list
reader = csv.reader(open("dataset2_inputs.txt", "rt"))
input_data=[]
for row in reader: 
   input_data.append([float(row[0])]) 

# Create a list of ones for the design matrix's first column
x_train = np.ones((len(input_data), 1))

# list to hold the sum of square of the models for different regularization lambda values
SSE_GLOBAL=[]

# create the design matrix for polynomial degree 20
for i in range(1,14):       
    x_raised_to_power = np.array([[pow(ele[0],i)] for ele in input_data])
    x_train = np.hstack((x_train,x_raised_to_power))

# Train the data with different lambdas for regularization variable 
for i in range(1,21):       
   # create the regularization variable
   alpha = pow(e,(-1 * i))
   alpha_matrix = alpha * np.identity(x_train.shape[1])

   # Train the model using the training data that we created with regularizer (i.e. get the weights)
   w = np.linalg.solve( (np.matmul(x_train.T , x_train) + alpha_matrix ), np.matmul(x_train.T , y_train))
   y_predict= np.matmul(x_train , w)

   # get sum of square of errors
   # divide the SSE by 2*N (according to slide 4 in lecture 6) 
   SSE = float( np.matmul ( (y_predict - y_train).T,(y_predict - y_train) ) / (2*len(x_train)))
   # add it to the global list
   SSE_GLOBAL.append(SSE)

plt.plot(list(range(-20,0)), SSE_GLOBAL[::-1], '-o', color='blue')
plt.xlabel('ln(lambda)')
plt.ylabel('SSE')
plt.xticks(list(range(-20,0)))
plt.show()




