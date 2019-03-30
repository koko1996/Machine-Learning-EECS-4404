import csv
import numpy as np
from random import randint
import matplotlib.pyplot as plt

data = np.genfromtxt(open("twodpoints.txt", "rb"), delimiter=",", dtype="float")

x,y = data.T
print data

plt.scatter(x,y)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
plt.show()