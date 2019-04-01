import csv
import numpy as np
import random 
import matplotlib.pyplot as plt

def init_hard_coded(k):
    return 0

def init_rand_uniform(input_data,k):
    centers = random.sample(list(input_data), k)
    return centers

def init_max_dist(input_data,k):
    return 0


def map_point_to_center(point,centers):
    min_dist = float("inf")
    center_id = -1 
    for i in range(0,len(centers)):
        # euclidian distance between point and centers[i]
        dist = np.linalg.norm(point-centers[i])
        if dist < min_dist:
            min_dist = dist
            center_id = i
    return center_id

# returns the distance
def dist_point_to_center(point,centers):
    min_dist = float("inf")
    center_id = -1 
    for i in range(0,len(centers)):
        # euclidian distance between point and centers[i]
        dist = np.linalg.norm(point-centers[i])
        if dist < min_dist:
            min_dist = dist
            center_id = i
    return min_dist


def cost(input_data, centers):
    cost = 0.0
    for point in input_data:
       cost += dist_point_to_center(point,centers)
    return cost

def update_centers(input_data,centers):
    new_centers = []
    return new_centers

data = np.genfromtxt(open("twodpoints.txt", "rb"), delimiter=",", dtype="float")

print (data)

centers = init_rand_uniform(data,3)


prev_cost = float("inf")
cur_cost = cost(data,centers)
while cur_cost < prev_cost:
    centers = update_centers(data,centers)
    prev_cost = cur_cost
    cur_cost = cost(data,centers)

clustered_data = []
for point in data:
    cluster_id = map_point_to_center(point,centers)
    clustered_point = point.append(cluster_id)
    clustered_data.append(clustered_point)

print (clustered_data)
