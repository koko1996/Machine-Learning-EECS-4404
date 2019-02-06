import csv
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random
from pprint import pprint


CHUNK_SIZE=10

reader = csv.reader(open("dataset1_outputs.txt", "rt"))
output_data=[]
for row in reader: 
   output_data.append([float(row[0])]) 

y_data=np.array(output_data)


reader = csv.reader(open("dataset1_inputs.txt", "rt"))
input_data=[]
for row in reader: 
   input_data.append([float(row[0])]) 

x_data = np.ones((len(input_data), 1))

model = linear_model.Ridge(alpha=0.001,normalize=False)

SSE_GLOBAL=[]

for i in range(1,21):        # for i in range(1,21):
    x_raised_to_power = np.array([[pow(ele[0],i)] for ele in input_data])
    x_data = np.hstack((x_data,x_raised_to_power))
    data = np.hstack((y_data,x_data))
    # print (data)
    # print (len(data))

    data= list(data)
    random.shuffle(data)
    
    # print()
    # print (data)
    # print (len(data))

    SSE=0
    for j in range(1,11):       
        train_data = data[: (CHUNK_SIZE * (j-1))]+ data[ (CHUNK_SIZE * j):]
        test_data = data[ (CHUNK_SIZE * (j-1)):(CHUNK_SIZE * j)]
        # print (train_data)
        # print (test_data)
        # print ("\n")

        x_train = np.array([ele[1:] for ele in train_data])
        y_train = np.array([[ele[0]] for ele in train_data])
        # print (x_train)
        # print (y_train)        

        x_test = np.array([ele[1:] for ele in test_data])
        y_test = np.array([[ele[0]] for ele in test_data])
        # print (x_test)
        # print (y_test)

        # Train the model using the training data that we created
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        SSE+= (mean_squared_error(y_test, y_pred) * len(x_test))
    SSE_GLOBAL.append(SSE/10)

print(SSE_GLOBAL)
plt.plot(list(range(1,21)), SSE_GLOBAL, '-o', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

