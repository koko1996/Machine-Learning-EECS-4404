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
erm_1 = linear_model.LinearRegression()
erm_5 = linear_model.LinearRegression()
erm_10 = linear_model.LinearRegression()
erm_20 = linear_model.LinearRegression()
# Train the models using the training data 
erm_1.fit(x_poly_1, y_train)
erm_5.fit(x_poly_5, y_train)
erm_10.fit(x_poly_10, y_train)
erm_20.fit(x_poly_20, y_train)

# plot the polynomials for erm models with the input data as dot plot
# where darker the color of the polynomial line the higher the degree of the polynomial
plt.plot(input_data, output_data, 'o', color='black')

plt.plot(X_plot, erm_1.intercept_ + X_plot*erm_1.coef_[1],color='#faff00')

plt.plot(X_plot, erm_5.intercept_ + pow(X_plot,1)*erm_5.coef_[1]  + pow(X_plot,2)*erm_5.coef_[2]  + pow(X_plot,3)*erm_5.coef_[3]  + pow(X_plot,4)*erm_5.coef_[4]  + pow(X_plot,5)*erm_5.coef_[5],color='#ffd000')

plt.plot(X_plot, erm_10.intercept_ + pow(X_plot,1)*erm_10.coef_[1]  + pow(X_plot,2)*erm_10.coef_[2]  + pow(X_plot,3)*erm_10.coef_[3]  + pow(X_plot,4)*erm_10.coef_[4]  + pow(X_plot,5)*erm_10.coef_[5] + pow(X_plot,6)*erm_10.coef_[6]  + pow(X_plot,7)*erm_10.coef_[7]  + pow(X_plot,8)*erm_10.coef_[8]  + pow(X_plot,9)*erm_10.coef_[9]  + pow(X_plot,10)*erm_10.coef_[10] ,color='#ff8800')

plt.plot(X_plot, erm_20.intercept_ + pow(X_plot,1)*erm_20.coef_[1]  + pow(X_plot,2)*erm_20.coef_[2]  + pow(X_plot,3)*erm_20.coef_[3]  + pow(X_plot,4)*erm_20.coef_[4]  + pow(X_plot,5)*erm_20.coef_[5] + pow(X_plot,6)*erm_20.coef_[6]  + pow(X_plot,7)*erm_20.coef_[7]  + pow(X_plot,8)*erm_20.coef_[8]  + pow(X_plot,9)*erm_20.coef_[9]  + pow(X_plot,10)*erm_20.coef_[10]  + pow(X_plot,11)*erm_20.coef_[11]  + pow(X_plot,12)*erm_20.coef_[12]  + pow(X_plot,13)*erm_20.coef_[13]  + pow(X_plot,14)*erm_20.coef_[14]  + pow(X_plot,15)*erm_20.coef_[15] + pow(X_plot,16)*erm_20.coef_[16]  + pow(X_plot,17)*erm_20.coef_[17]  + pow(X_plot,18)*erm_20.coef_[18]  + pow(X_plot,19)*erm_20.coef_[19]  + pow(X_plot,20)*erm_20.coef_[20] ,color='#ff0000')

plt.title('ERM')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# create the rlm models for polynomial degree of 1, 5, 10 and 20 with regularization (where lambda = 0.001)
rlm_1 = linear_model.Ridge(alpha=0.001,normalize=False)
rlm_5 = linear_model.Ridge(alpha=0.001,normalize=False)
rlm_10 = linear_model.Ridge(alpha=0.001,normalize=False)
rlm_20 = linear_model.Ridge(alpha=0.001,normalize=False)
# Train the rlm models using the training data 
rlm_1.fit(x_poly_1, y_train)
rlm_5.fit(x_poly_5, y_train)
rlm_10.fit(x_poly_10, y_train)
rlm_20.fit(x_poly_20, y_train)


# plot the polynomials for rlm models with the input data as dot plot
# where darker the color of the polynomial line the higher the degree of the polynomial
plt.plot(input_data, output_data, 'o', color='black')

plt.plot(X_plot, rlm_1.intercept_ + X_plot*rlm_1.coef_[1],color='#faff00')

plt.plot(X_plot, rlm_5.intercept_ + pow(X_plot,1)*rlm_5.coef_[1]  + pow(X_plot,2)*rlm_5.coef_[2]  + pow(X_plot,3)*rlm_5.coef_[3]  + pow(X_plot,4)*rlm_5.coef_[4]  + pow(X_plot,5)*rlm_5.coef_[5],color='#ffd000')

plt.plot(X_plot, rlm_10.intercept_ + pow(X_plot,1)*rlm_10.coef_[1]  + pow(X_plot,2)*rlm_10.coef_[2]  + pow(X_plot,3)*rlm_10.coef_[3]  + pow(X_plot,4)*rlm_10.coef_[4]  + pow(X_plot,5)*rlm_10.coef_[5] + pow(X_plot,6)*rlm_10.coef_[6]  + pow(X_plot,7)*rlm_10.coef_[7]  + pow(X_plot,8)*rlm_10.coef_[8]  + pow(X_plot,9)*rlm_10.coef_[9]  + pow(X_plot,10)*rlm_10.coef_[10] ,color='#ff8800')

plt.plot(X_plot, rlm_20.intercept_ + pow(X_plot,1)*rlm_20.coef_[1]  + pow(X_plot,2)*rlm_20.coef_[2]  + pow(X_plot,3)*rlm_20.coef_[3]  + pow(X_plot,4)*rlm_20.coef_[4]  + pow(X_plot,5)*rlm_20.coef_[5] + pow(X_plot,6)*rlm_20.coef_[6]  + pow(X_plot,7)*rlm_20.coef_[7]  + pow(X_plot,8)*rlm_20.coef_[8]  + pow(X_plot,9)*rlm_20.coef_[9]  + pow(X_plot,10)*rlm_20.coef_[10]  + pow(X_plot,11)*rlm_20.coef_[11]  + pow(X_plot,12)*rlm_20.coef_[12]  + pow(X_plot,13)*rlm_20.coef_[13]  + pow(X_plot,14)*rlm_20.coef_[14]  + pow(X_plot,15)*rlm_20.coef_[15] + pow(X_plot,16)*rlm_20.coef_[16]  + pow(X_plot,17)*rlm_20.coef_[17]  + pow(X_plot,18)*rlm_20.coef_[18]  + pow(X_plot,19)*rlm_20.coef_[19]  + pow(X_plot,20)*rlm_20.coef_[20] ,color='#ff0000')

plt.title('RLM alpha=0.001')
plt.xlabel('x')
plt.ylabel('y')
plt.show()




rlm_1 = linear_model.Ridge(alpha=0.1,normalize=False)
rlm_5 = linear_model.Ridge(alpha=0.1,normalize=False)
rlm_10 = linear_model.Ridge(alpha=0.1,normalize=False)
rlm_20 = linear_model.Ridge(alpha=0.1,normalize=False)
# Train the model using the training data 
rlm_1.fit(x_poly_1, y_train)
rlm_5.fit(x_poly_5, y_train)
rlm_10.fit(x_poly_10, y_train)
rlm_20.fit(x_poly_20, y_train)

plt.plot(input_data, output_data, 'o', color='black')

plt.plot(X_plot, rlm_1.intercept_ + X_plot*rlm_1.coef_[1],color='#faff00')

plt.plot(X_plot, rlm_5.intercept_ + pow(X_plot,1)*rlm_5.coef_[1]  + pow(X_plot,2)*rlm_5.coef_[2]  + pow(X_plot,3)*rlm_5.coef_[3]  + pow(X_plot,4)*rlm_5.coef_[4]  + pow(X_plot,5)*rlm_5.coef_[5],color='#ffd000')

plt.plot(X_plot, rlm_10.intercept_ + pow(X_plot,1)*rlm_10.coef_[1]  + pow(X_plot,2)*rlm_10.coef_[2]  + pow(X_plot,3)*rlm_10.coef_[3]  + pow(X_plot,4)*rlm_10.coef_[4]  + pow(X_plot,5)*rlm_10.coef_[5] + pow(X_plot,6)*rlm_10.coef_[6]  + pow(X_plot,7)*rlm_10.coef_[7]  + pow(X_plot,8)*rlm_10.coef_[8]  + pow(X_plot,9)*rlm_10.coef_[9]  + pow(X_plot,10)*rlm_10.coef_[10] ,color='#ff8800')

plt.plot(X_plot, rlm_20.intercept_ + pow(X_plot,1)*rlm_20.coef_[1]  + pow(X_plot,2)*rlm_20.coef_[2]  + pow(X_plot,3)*rlm_20.coef_[3]  + pow(X_plot,4)*rlm_20.coef_[4]  + pow(X_plot,5)*rlm_20.coef_[5] + pow(X_plot,6)*rlm_20.coef_[6]  + pow(X_plot,7)*rlm_20.coef_[7]  + pow(X_plot,8)*rlm_20.coef_[8]  + pow(X_plot,9)*rlm_20.coef_[9]  + pow(X_plot,10)*rlm_20.coef_[10]  + pow(X_plot,11)*rlm_20.coef_[11]  + pow(X_plot,12)*rlm_20.coef_[12]  + pow(X_plot,13)*rlm_20.coef_[13]  + pow(X_plot,14)*rlm_20.coef_[14]  + pow(X_plot,15)*rlm_20.coef_[15] + pow(X_plot,16)*rlm_20.coef_[16]  + pow(X_plot,17)*rlm_20.coef_[17]  + pow(X_plot,18)*rlm_20.coef_[18]  + pow(X_plot,19)*rlm_20.coef_[19]  + pow(X_plot,20)*rlm_20.coef_[20] ,color='#ff0000')

plt.title('RLM alpha=0.1')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

