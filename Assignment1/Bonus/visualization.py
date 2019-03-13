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
from pprint import pprint

# define the vaules on the x axis 
X_plot = np.linspace(-1,1,100)

# Read output data as floating point integers to the list
reader = csv.reader(open("dataset2_outputs.txt", "rt"))
output_data=[]
for row in reader: 
   output_data.append(float(row[0])) 
# convert the list to np array
y_train=np.array(output_data)

# Read Input data as floating point integers to the list
reader = csv.reader(open("dataset2_inputs.txt", "rt"))
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
for i in range(11,14):
    x_raised_to_power = np.array([[pow(ele[0],i)] for ele in input_data])
    x_poly_20 = np.hstack((x_poly_20,x_raised_to_power))


alpha0 = 0.000

w_rlm_10 = np.linalg.solve( (np.matmul(x_poly_10.T , x_poly_10) + (alpha0 * np.identity(x_poly_10.shape[1])) ), np.matmul(x_poly_10.T , y_train))


# plot the polynomials for rlm models with the input data as dot plot
# where darker the color of the polynomial line the higher the degree of the polynomial
plt.plot(input_data, output_data, 'o', color='black')

plt.plot(X_plot, w_rlm_10[0] + pow(X_plot,1)*w_rlm_10[1]  + pow(X_plot,2)*w_rlm_10[2]  + pow(X_plot,3)*w_rlm_10[3]  + pow(X_plot,4)*w_rlm_10[4]  + pow(X_plot,5)*w_rlm_10[5] + pow(X_plot,6)*w_rlm_10[6]  + pow(X_plot,7)*w_rlm_10[7]  + pow(X_plot,8)*w_rlm_10[8]  + pow(X_plot,9)*w_rlm_10[9]  + pow(X_plot,10)*w_rlm_10[10] ,color='#ff8800')



plt.title('RLM alpha=0.001')
plt.xlabel('x')
plt.ylabel('y')
plt.show()