###################################################################################################
# EECS4404 Assignment 1                                                                           #
# Filename: cross_validation.py                                                                   #   
# Author: NANAH JI, KOKO                                                                          #
# Email: koko96@my.yorku.com                                                                      #
###################################################################################################

import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# the chunk size for each fold in the k fold algorithm
CHUNK_SIZE=10

# Read output data as floating point integers to the list
reader = csv.reader(open("dataset1_outputs.txt", "rt"))
output_data=[]
for row in reader: 
   output_data.append([float(row[0])]) 

# convert the list to np array
y_data=np.array(output_data)

# Read Input data as floating point integers to the list
reader = csv.reader(open("dataset1_inputs.txt", "rt"))
input_data=[]
for row in reader: 
   input_data.append([float(row[0])]) 

# Create a list of ones for the design matrix's first column
x_data = np.ones((len(input_data), 1))

# create linear model without regularization
model = linear_model.LinearRegression()

# list to hold the average sum of square of test errors for each model that are polynomial with different degrees  
SSE_GLOBAL=[]

# for each degree of polynomial from 1 to 20 fit erm model and calculate the average of the sum of square of test errors
for i in range(1,21):      
    # create the column next column in the design matrix by raising the input data to the ith power
    x_raised_to_power = np.array([[pow(ele[0],i)] for ele in input_data])
    # add the new column to the design matrix
    x_data = np.hstack((x_data,x_raised_to_power))
    # add the y vector to the design matrix so that shuffling will not mess things up
    data = np.hstack((y_data,x_data))

    # conver the data matrix to list of lists to make the shuffle work
    data= list(data)
    # shuffle the data to do the kfold
    random.shuffle(data)

    SSE=0      # variable to accumulate sum of square of test errors
    for j in range(1,11):       
        # devide the data to test and train sets 
        train_data = data[: (CHUNK_SIZE * (j-1))]+ data[ (CHUNK_SIZE * j):]
        test_data = data[ (CHUNK_SIZE * (j-1)):(CHUNK_SIZE * j)]

        # extract the x vector from the train data
        x_train = np.array([ele[1:] for ele in train_data])
        # extract the y vector from the train data 
        y_train = np.array([[ele[0]] for ele in train_data])      

        # extract the x vector from the test data
        x_test = np.array([ele[1:] for ele in test_data])
        # extract the y vector from the tests data 
        y_test = np.array([[ele[0]] for ele in test_data])

        # Train the model using the training data that was chosen
        model.fit(x_train, y_train)
        # test the data on the test set
        y_pred = model.predict(x_test)
        # add the sum of square of errors of the testing the test set to the running accumulator
        SSE+= (mean_squared_error(y_test, y_pred) * len(x_test))
    # add the average sum of square of errors of tests to the global list
    SSE_GLOBAL.append(SSE/10)

plt.plot(list(range(1,21)), SSE_GLOBAL, '-o', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks(list(range(1,21)))
plt.show()

