###################################################################################################
# EECS4404 Assignment 1                                                                           #
# Filename: visualization.py                                                                      #   
# Author: NANAH JI, KOKO                                                                          #
# Email: koko96@my.yorku.com                                                                      #
###################################################################################################

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# define the vaules on the x axis 
X_plot = np.linspace(-1,1,100)

# Read output data as floating point integers to the list
reader = csv.reader(open("dataset1_outputs.txt", "rt"))
output_data=[]
for row in reader: 
   output_data.append(float(row[0])) 
# convert the list to np array
y_train=np.array(output_data)

# Read Input data as floating point integers to the list
reader = csv.reader(open("dataset1_inputs.txt", "rt"))
input_data=[]
for row in reader: 
   input_data.append([float(row[0])]) 
# Create a list of ones for the design matrix's first column
x_train = np.ones((len(input_data), 1))

# create design matrix for polynomial of degree one 
x_poly_1 = np.hstack((x_train,input_data))

# create design matrix for polynomial of degree 5
x_poly_5 = x_poly_1
for i in range(2,6):
    x_raised_to_power = np.array([[pow(ele[0],i)] for ele in input_data])
    x_poly_5 = np.hstack((x_poly_5,x_raised_to_power))

# create design matrix for polynomial of degree 10
x_poly_10 = x_poly_5
for i in range(6,11):
    x_raised_to_power = np.array([[pow(ele[0],i)] for ele in input_data])
    x_poly_10 = np.hstack((x_poly_10,x_raised_to_power))

# create design matrix for polynomial of degree 20
x_poly_20 = x_poly_10
for i in range(11,21):
    x_raised_to_power = np.array([[pow(ele[0],i)] for ele in input_data])
    x_poly_20 = np.hstack((x_poly_20,x_raised_to_power))

# create the erm models for polynomial degree of 1, 5, 10 and 20
# Train the models using the training data 
w_erm_1 =  np.linalg.solve(np.matmul(x_poly_1.T , x_poly_1), np.matmul(x_poly_1.T , y_train))
w_erm_5 =  np.linalg.solve(np.matmul(x_poly_5.T , x_poly_5), np.matmul(x_poly_5.T , y_train))
w_erm_10 = np.linalg.solve(np.matmul(x_poly_10.T , x_poly_10), np.matmul(x_poly_10.T , y_train))
w_erm_20 = np.linalg.solve(np.matmul(x_poly_20.T , x_poly_20), np.matmul(x_poly_20.T , y_train))

# plot the polynomials for erm models with the input data as dot plot
# where darker the color of the polynomial line the higher the degree of the polynomial
plt.plot(input_data, output_data, 'o', color='black')

plt.plot(X_plot, w_erm_1[0] + X_plot*w_erm_1[1],color='#faff00')

plt.plot(X_plot, w_erm_5[0] + pow(X_plot,1)*w_erm_5[1]  + pow(X_plot,2)*w_erm_5[2]  + pow(X_plot,3)*w_erm_5[3]  + pow(X_plot,4)*w_erm_5[4]  + pow(X_plot,5)*w_erm_5[5],color='#ffd000')

plt.plot(X_plot, w_erm_10[0] + pow(X_plot,1)*w_erm_10[1]  + pow(X_plot,2)*w_erm_10[2]  + pow(X_plot,3)*w_erm_10[3]  + pow(X_plot,4)*w_erm_10[4]  + pow(X_plot,5)*w_erm_10[5] + pow(X_plot,6)*w_erm_10[6]  + pow(X_plot,7)*w_erm_10[7]  + pow(X_plot,8)*w_erm_10[8]  + pow(X_plot,9)*w_erm_10[9]  + pow(X_plot,10)*w_erm_10[10] ,color='#ff8800')

plt.plot(X_plot, w_erm_20[0] + pow(X_plot,1)*w_erm_20[1]  + pow(X_plot,2)*w_erm_20[2]  + pow(X_plot,3)*w_erm_20[3]  + pow(X_plot,4)*w_erm_20[4]  + pow(X_plot,5)*w_erm_20[5] + pow(X_plot,6)*w_erm_20[6]  + pow(X_plot,7)*w_erm_20[7]  + pow(X_plot,8)*w_erm_20[8]  + pow(X_plot,9)*w_erm_20[9]  + pow(X_plot,10)*w_erm_20[10]  + pow(X_plot,11)*w_erm_20[11]  + pow(X_plot,12)*w_erm_20[12]  + pow(X_plot,13)*w_erm_20[13]  + pow(X_plot,14)*w_erm_20[14]  + pow(X_plot,15)*w_erm_20[15] + pow(X_plot,16)*w_erm_20[16]  + pow(X_plot,17)*w_erm_20[17]  + pow(X_plot,18)*w_erm_20[18]  + pow(X_plot,19)*w_erm_20[19]  + pow(X_plot,20)*w_erm_20[20] ,color='#ff0000')

plt.title('ERM')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


alpha = 0.001

# create the rlm models for polynomial degree of 1, 5, 10 and 20 with regularization (where lambda = 0.001)
w_rlm_1 =  np.linalg.solve( (np.matmul(x_poly_1.T , x_poly_1) + (alpha * np.identity(x_poly_1.shape[1])) ), np.matmul(x_poly_1.T , y_train))
w_rlm_5 =  np.linalg.solve( (np.matmul(x_poly_5.T , x_poly_5) + (alpha * np.identity(x_poly_5.shape[1])) ), np.matmul(x_poly_5.T , y_train))
w_rlm_10 = np.linalg.solve( (np.matmul(x_poly_10.T , x_poly_10) + (alpha * np.identity(x_poly_10.shape[1])) ), np.matmul(x_poly_10.T , y_train))
w_rlm_20 = np.linalg.solve( (np.matmul(x_poly_20.T , x_poly_20) + (alpha * np.identity(x_poly_20.shape[1])) ), np.matmul(x_poly_20.T , y_train))


# plot the polynomials for rlm models with the input data as dot plot
# where darker the color of the polynomial line the higher the degree of the polynomial
plt.plot(input_data, output_data, 'o', color='black')

plt.plot(X_plot, w_rlm_1[0] + X_plot*w_rlm_1[1],color='#faff00')

plt.plot(X_plot, w_rlm_5[0] + pow(X_plot,1)*w_rlm_5[1]  + pow(X_plot,2)*w_rlm_5[2]  + pow(X_plot,3)*w_rlm_5[3]  + pow(X_plot,4)*w_rlm_5[4]  + pow(X_plot,5)*w_rlm_5[5],color='#ffd000')

plt.plot(X_plot, w_rlm_10[0] + pow(X_plot,1)*w_rlm_10[1]  + pow(X_plot,2)*w_rlm_10[2]  + pow(X_plot,3)*w_rlm_10[3]  + pow(X_plot,4)*w_rlm_10[4]  + pow(X_plot,5)*w_rlm_10[5] + pow(X_plot,6)*w_rlm_10[6]  + pow(X_plot,7)*w_rlm_10[7]  + pow(X_plot,8)*w_rlm_10[8]  + pow(X_plot,9)*w_rlm_10[9]  + pow(X_plot,10)*w_rlm_10[10] ,color='#ff8800')

plt.plot(X_plot, w_rlm_20[0] + pow(X_plot,1)*w_rlm_20[1]  + pow(X_plot,2)*w_rlm_20[2]  + pow(X_plot,3)*w_rlm_20[3]  + pow(X_plot,4)*w_rlm_20[4]  + pow(X_plot,5)*w_rlm_20[5] + pow(X_plot,6)*w_rlm_20[6]  + pow(X_plot,7)*w_rlm_20[7]  + pow(X_plot,8)*w_rlm_20[8]  + pow(X_plot,9)*w_rlm_20[9]  + pow(X_plot,10)*w_rlm_20[10]  + pow(X_plot,11)*w_rlm_20[11]  + pow(X_plot,12)*w_rlm_20[12]  + pow(X_plot,13)*w_rlm_20[13]  + pow(X_plot,14)*w_rlm_20[14]  + pow(X_plot,15)*w_rlm_20[15] + pow(X_plot,16)*w_rlm_20[16]  + pow(X_plot,17)*w_rlm_20[17]  + pow(X_plot,18)*w_rlm_20[18]  + pow(X_plot,19)*w_rlm_20[19]  + pow(X_plot,20)*w_rlm_20[20] ,color='#ff0000')

plt.title('RLM alpha=0.001')
plt.xlabel('x')
plt.ylabel('y')
plt.show()