import csv
import matplotlib.pyplot as plt

# Read output data
reader = csv.reader(open("dataset1_outputs.txt", "rt"))
output_data=[]
for row in reader: 
   output_data.append(float(row[0]))

# Read Input data
reader = csv.reader(open("dataset1_inputs.txt", "rt"))
input_data=[]
for row in reader: 
   input_data.append(float(row[0]))  

# Plot the data as dot plot
plt.plot(input_data, output_data, 'o', color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
