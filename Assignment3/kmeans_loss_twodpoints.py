import csv
import numpy as np
import random 
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
        cost += np.linalg.norm(point[:-1]-centers[int(point[-1])])
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

# read data
init_data = np.genfromtxt(open("twodpoints.txt", "rb"), delimiter=",", dtype="float")


costs=[]
# calculate the cost for clustering k-means where k is from 1 to 10
for k in range(1,11):
    number_of_clusters=k
    # initialize the centers from original data
    init_centers = init_max_dist(init_data,number_of_clusters)
    centers = init_centers

    # append the column that identifies the clusters that each point belongs to
    zeros = np.zeros((len(init_data),1), dtype=float)
    data = np.append(init_data,zeros ,axis=1)

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

    # add the current cost for k to the list of costs
    costs.append(cur_cost)
    # plt.scatter(data[:,0],data[:,1],c=data[:,2])
    # plt.scatter(init_centers[:,0],init_centers[:,1], c="black")
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()

# plot the costs 
plt.plot(list(range(1,len(costs)+1)), costs,color='blue')
plt.xlabel('number of clusters')
plt.ylabel('Cost')
plt.title('Kmeans cost')
plt.show()


