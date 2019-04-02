import csv
import numpy as np
import random 
import collections
import matplotlib.pyplot as plt


# initialize centers to k data points starts by picking the 
# first center uniformly at random,and then chooses each next
# center to be the datapoint that maximizes the sum of (euclidean)
# distances from the previous selected center points (the other centers)
def init_max_dist(input_data,k):
    centers=[]      
    # select random index 
    index = random.randint(0, len(input_data)-1)
    centers.append(input_data[index])
    
    # pick the next k-1 centers
    for i in range(0,k-1):
        next_center = np.zeros((1,input_data.shape[1]), dtype=float)
        max_total_dist=-1
        # append the point with maximum distance from the center points in centers
        for point in input_data:
            total_dist=0
            for c in centers:
                total_dist += np.linalg.norm(point- c)
            if total_dist > max_total_dist:
                max_total_dist = total_dist
                next_center = point
        centers.append(next_center)
    return np.array(centers)

# returns the sum of kmeans cost of each input data
def cost(input_data, centers):
    cost = 0.0
    for point in input_data:
        # dist from point to it's center
        cost += (np.linalg.norm(point[:-1]-centers[int(point[-1])]) ** 2)
    return cost

# returns the new centers for each cluster by computing the mean of every cluster as a newcenter
def update_centers(input_data,number_of_centers):
    # initialze to zeros
    new_centers = np.zeros([number_of_centers,input_data.shape[1]-1])
    count_per_center =  np.zeros([number_of_centers,1])

    # calulate the total sum and count for each cluster
    for point in input_data:
        new_centers[int(point[-1])] += point[:-1]
        count_per_center[int(point[-1])] += 1.0
    
    # Get the average point of each cluster
    for i in range(0,len(new_centers)):
        new_centers[i] = new_centers[i] * (1/max(count_per_center[i],1))
    return new_centers

# maps a datapoint to it's closest center
def map_point_to_center(point,centers):
    min_dist = float("inf")
    center_id = -1 
    for i in range(0,len(centers)):
        # euclidian distance between point and centers[i]
        dist = np.linalg.norm(point[:-1]-centers[i])
        if dist < min_dist:
            min_dist = dist
            center_id = i
    return center_id

# changes the input_data by updating the clusters that each point belongs to
def update_clusters(input_data,centers):
    for point in input_data:
        point[-1]=map_point_to_center(point,centers)
    return input_data


# main method

### read and preprocess the data
with open('seeds_dataset.txt') as f:
    reader = csv.reader(f, delimiter='\t')
    rows = list(reader)
data = []
# preprocess the data by removing empty elements and      
for element in rows:
    next_row =  [i for i in element if i != '']
    data.append(next_row)
np.set_printoptions(threshold=np.inf)
data=np.array([np.array(xi,dtype=float) for xi in data])  
original_clustering=data[:,-1]
data= data[:,:-1]


number_of_clusters=3
# initialize the centers from original data
init_centers = init_max_dist(data,number_of_clusters)
centers = init_centers

# append the column that identifies the clusters that each point belongs to
zeros = np.zeros((len(data),1), dtype=float)
data = np.append(data,zeros ,axis=1)

# kmeans algorithm
# calulate the best center points for each cluster
# stop when convereged 
prev_cost = float("inf")
data = update_clusters(data,centers)
cur_cost = cost(data,centers)
while cur_cost < prev_cost:
    centers = update_centers(data,number_of_clusters)
    data = update_clusters(data,centers)
    prev_cost = cur_cost
    cur_cost = cost(data,centers)

# final_clustering indicates the cluster assignments of the points
# adding one because the question asks for clusters from 1 to k
# but the above code is written in a way (which makes it easier)
# that assigns cluster from 0 to k-1
final_clustering= [int(x)+1 for x in data[:,-1]]

# calculate the binary loss overall data
# empirical_loss = np.count_nonzero(final_clustering - original_clustering) / float(len(data))
# print(empirical_loss)


# size of each cluster since the all of them have the same size in this case
chunk_size = len(original_clustering) / number_of_clusters

# figure out the id that matches cluster id 1 in the original data
counts = collections.Counter(final_clustering[:chunk_size])
new_list = sorted(final_clustering[:chunk_size], key=counts.get, reverse=True) 
print(new_list)
values = new_list
clust_id_1 = values[0] 
print (clust_id_1) 

# figure out the id that matches cluster id 2 in the original data
counts = collections.Counter(final_clustering[chunk_size:2*chunk_size])
new_list = sorted(final_clustering[chunk_size:2*chunk_size], key=counts.get, reverse=True) 
print(new_list)
values = new_list

clust_id_2 = 0
for i in values:
    if i != clust_id_1:
        clust_id_2 = i
print (clust_id_2) 


# figure outt the id that matches cluster id 3 in the original data
counts = collections.Counter(final_clustering[2*chunk_size:3*chunk_size])
new_list = sorted(final_clustering[2*chunk_size:3*chunk_size], key=counts.get, reverse=True) 
print(new_list)
values = new_list
clust_id_3 = 0
for i in values:
    if i != clust_id_1 and i != clust_id_2:
        clust_id_3 = i
print (clust_id_3) 

# modify the cluster ids 
modified_final_clustering = []
for i in final_clustering:
    if i == clust_id_1:
        modified_final_clustering.append(1)
    elif i == clust_id_2:
        modified_final_clustering.append(2)
    else :
        modified_final_clustering.append(3)

empirical_loss = np.count_nonzero(modified_final_clustering - original_clustering) / float(len(data))
print(empirical_loss)