###################################################################################################
# EECS4404 Assignment 1                                                                           #
# Filename: ERM.py                                                                                #   
# Author: NANAH JI, KOKO                                                                          #
# Email: koko96@my.yorku.com                                                                      #
###################################################################################################

import csv
import numpy as np
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

# list to hold the sum of square of the models for different degrees of polynomial of the regression model
SSE_GLOBAL=[]

# for each degree of polynomial from 1 to 20 fit erm model and calculate the sum of square of errors
for i in range(1,21):      
   # augment the design matrix with one additional column for the degree of polynomial i
   x_raised_to_power = np.array([[pow(ele[0],i)] for ele in input_data])
   x_train = np.hstack((x_train,x_raised_to_power))
   
   # Train the model using the training data that we created (i.e get the weights)
   w = np.linalg.solve(np.matmul(x_train.T , x_train), np.matmul(x_train.T , y_train))
   if i == 10 :
      w = np.array([   1.39978297,    3.66085768,  -35.32611838,  -30.10042907,162.62229905,  118.23231035, -299.66018772, -174.93062342,263.6970981 ,   83.51178451,  -93.4279917 ])
      print(w)
   # predict y for each input (row)
   y_predict= np.matmul(x_train , w)

   # get sum of square of errors
   # divide the SSE by 2*N (according to slide 4 in lecture 6) 
   SSE = float(np.matmul ( (y_predict - y_train).T,(y_predict - y_train) ) /(2*len(x_train)) )
   #  add it to the global list
   SSE_GLOBAL.append(SSE)


plt.plot(list(range(1,21)), SSE_GLOBAL, '-o', color='blue')
plt.xlabel('W')
plt.ylabel('SSE')
plt.xticks(list(range(1,21)))
plt.show()


