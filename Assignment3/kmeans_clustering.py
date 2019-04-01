import csv
import numpy as np
import random 
import matplotlib.pyplot as plt

def init_hard_coded(input_data,k):
    return np.array([[-1.5,4.5],[3.5,4.5],[0,-4]])

def init_rand_uniform(input_data,k):
    centers = random.sample(list(input_data), k)
    return np.array(centers)

def init_max_dist(input_data,k):
    centers=[]
    index = random.randint(0, len(input_data)-1)
    centers.append(input_data[index])
    
    for i in range(0,k-1):
        next_center = np.zeros((1,input_data.shape[1]), dtype=float)
        max_total_dist=-1
        for point in input_data:
            total_dist=0
            for c in centers:
                total_dist += np.linalg.norm(point- c)
            if total_dist > max_total_dist:
                max_total_dist = total_dist
                next_center = point
        centers.append(next_center)
    return np.array(centers)


def cost(input_data, centers):
    cost = 0.0
    for point in input_data:
        # dist from point to it's center
        cost += np.linalg.norm(point[:-1]-centers[int(point[-1])])
    return cost

def update_centers(input_data,number_of_centers):
    new_centers = np.zeros([number_of_centers,input_data.shape[1]-1])
    count_per_center =  np.zeros([number_of_centers,1])

    for point in input_data:
        new_centers[int(point[-1])] += point[:-1]
        count_per_center[int(point[-1])] += 1.0

    for i in range(0,len(new_centers)):
        new_centers[i] = new_centers[i] * (1/max(count_per_center[i],1))
    print("number_of_centers")
    print(count_per_center)
    return new_centers


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


def update_clusters(input_data,centers):
    for point in input_data:
        point[-1]=map_point_to_center(point,centers)
    return input_data



number_of_clusters=4
data = np.genfromtxt(open("twodpoints.txt", "rb"), delimiter=",", dtype="float")
init_centers = init_max_dist(data,number_of_clusters)
centers = init_centers

zeros = np.zeros((len(data),1), dtype=float)
data = np.append(data,zeros ,axis=1)


prev_cost = float("inf")
data = update_clusters(data,centers)
cur_cost = cost(data,centers)

while cur_cost < prev_cost:
    centers = update_centers(data,number_of_clusters)
    data = update_clusters(data,centers)

    prev_cost = cur_cost
    cur_cost = cost(data,centers)

print(data)
print(centers)
print(init_centers)

x,y = init_centers.T

plt.scatter(data[:,0],data[:,1],c=data[:,2])
plt.scatter(init_centers[:,0],init_centers[:,1], c="black")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

