import csv
import numpy as np
from random import randint
import matplotlib.pyplot as plt

# read the data
data = np.genfromtxt(open("twodpoints.txt", "rb"), delimiter=",", dtype="float")

# extract the columns
x,y = data.T
print (data) 

# plot the data
plt.scatter(x,y)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()