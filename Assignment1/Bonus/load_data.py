###################################################################################################
# EECS4404 Assignment 1                                                                           #
# Filename: load_data.py                                                                          #
# Author: NANAH JI, KOKO                                                                          #
# Email: koko96@my.yorku.com                                                                      #
###################################################################################################

import csv
import numpy as np
import matplotlib.pyplot as plt

# Read output data
reader = csv.reader(open("../dataset2_outputs.txt", "rt"))
output_data=[]
for row in reader: 
   output_data.append(float(row[0]))

# Read Input data
reader = csv.reader(open("../dataset2_inputs.txt", "rt"))
input_data=[]
for row in reader: 
   input_data.append(float(row[0]))  

# Plot the data as dot plot
w_rlm_10 = np.array([   1.60106506,    1.95169855,  -35.14402046,    0.32930283,156.0779382 ,    0.98422534, -260.76373712,  -11.45070446,189.22383199,    7.21387434,  -50.42243347])
X_plot = np.linspace(-1,1,100)
actual = np.array([   0.90606798,    2.52677409,  -23.25928251,   -7.57049332, 103.96722607,   32.95231045, -177.63127309,  -60.10981207,143.11419681,   32.11332213,  -47.06355137])

plt.plot(X_plot, w_rlm_10[0] + pow(X_plot,1)*w_rlm_10[1]  + pow(X_plot,2)*w_rlm_10[2]  + pow(X_plot,3)*w_rlm_10[3]  + pow(X_plot,4)*w_rlm_10[4]  + pow(X_plot,5)*w_rlm_10[5] + pow(X_plot,6)*w_rlm_10[6]  + pow(X_plot,7)*w_rlm_10[7]  + pow(X_plot,8)*w_rlm_10[8]  + pow(X_plot,9)*w_rlm_10[9]  + pow(X_plot,10)*w_rlm_10[10] ,color='#ff8800')

plt.plot(X_plot, actual[0] + pow(X_plot,1)*actual[1]  + pow(X_plot,2)*actual[2]  + pow(X_plot,3)*actual[3]  + pow(X_plot,4)*actual[4]  + pow(X_plot,5)*actual[5] + pow(X_plot,6)*actual[6]  + pow(X_plot,7)*actual[7]  + pow(X_plot,8)*actual[8]  + pow(X_plot,9)*actual[9]  + pow(X_plot,10)*actual[10] ,color='blue')
plt.plot(input_data, output_data, 'o', color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
