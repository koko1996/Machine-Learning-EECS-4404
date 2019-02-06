###################################################################################################
# EECS4404 Assignment 1                                                                           #
# Filename: ERM.py                                                                                #   
# Author: NANAH JI, KOKO                                                                          #
# Email: koko96@my.yorku.com                                                                      #
###################################################################################################

import csv
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read output data as floating point integers to the list
reader = csv.reader(open("dataset1_outputs.txt", "rt"))
output_data=[]
for row in reader: 
   output_data.append(float(row[0])) 

# convert the list to np array
y_train=np.array(output_data)

# Read input data as floating point integers to the list
reader = csv.reader(open("dataset1_inputs.txt", "rt"))
input_data=[]
for row in reader: 
   input_data.append([float(row[0])]) 

# Create a list of ones for the design matrix's first column
x_train = np.ones((len(input_data), 1))

# create linear model without regularization
model = linear_model.LinearRegression()

# list to hold the sum of square of the models for different degrees of polynomial of the regression model
SSE_GLOBAL=[]

# for each degree of polynomial from 1 to 20 fit erm model and calculate the sum of square of errors
for i in range(1,21):      
    x_raised_to_power = np.array([[pow(ele[0],i)] for ele in input_data])
    x_train = np.hstack((x_train,x_raised_to_power))
    # Train the model using the training data that we created
    model.fit(x_train, y_train)

    SSE=0
    # calculation of sum of square of errors for the model 
    for x,y_true in zip(x_train,output_data):
        x_pred = np.array([x])
        y_pred = model.predict(x_pred)[0]
        SSE +=  pow((y_pred- y_true),2)
    SSE_GLOBAL.append(SSE)


plt.plot(list(range(1,21)), SSE_GLOBAL, '-o', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks(list(range(1,21)))
plt.show()


