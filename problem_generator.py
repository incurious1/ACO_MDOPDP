from dis import dis
import numpy as np
from scipy.spatial import distance_matrix
import random

def generate_random_samples(n_cus, n_vehicles, scale=100, road_scaling=True, noise=True):
    '''
    generate point coordinates, distance matrix, and pickups-deliveries pairs from the given numbers of customers and vehicles

    Parameters:
        n_cus (int): number of customers (even number)
        n_vehicles (int): number of available vehicle
        scale (int/float): scale of the (map) coordinates compared to a unit square
        road_scaling (bool): road scaling factor, replicate detour, u-turn, etc.
        noise (bool): add noise to the distance between every pairs of points (positive)

    Returns:
        distance_mat (ndarray): distance matrix size n*n (n = n_cus + n_vehicles)
        pickups_delivery_list (list of pairs): list of pickups-deliveries pairs [pickup, delivery]
        coords (ndarray): coordinates of points size n*2 
        pickups (list size: num_vehicles/2): list of pickup nodes
        deliveries (list size: num_vehicles/2): list of delivery nodes
    '''
    n_nodes = n_cus + n_vehicles
    # generate node coordinates 
    coords = np.round(((np.random.rand(n_nodes, 2) - 0.5) * 2) * scale, 3)
    distance_mat = np.array(distance_matrix(coords, coords))
    # road scaling factor 
    if road_scaling:
        distance_mat += distance_mat*(0.1*scale)**(-(distance_mat/scale)**2) 
    # add noise
    if noise:
        noise = abs(np.random.normal(0, scale*0.1, (n_nodes)**2)).reshape(n_nodes, n_nodes) 
        np.fill_diagonal(noise,0)
        distance_mat += noise
    distance_mat = np.rint(distance_mat)
    # generate pickups-deliveries pairs
    node_list = [*range(n_vehicles, n_nodes)]
    pickups_delivery_list = []
    pickups = []
    deliveries = []
    for _ in range(n_cus//2):
        rand_sample = random.sample(node_list, 2)
        pickups_delivery_list.append(rand_sample)
        pickups.append(rand_sample[0])
        deliveries.append(rand_sample[1])
        node_list = [e for e in node_list if e not in rand_sample]

    return distance_mat, pickups_delivery_list, coords, pickups, deliveries

def generate_demands(pickups, deliveries, n_vehicle, weighted=True, weight_range=(20,100)):
    '''
    generate demands for each nodes 

    Parameters:
        pickups (list size: num_vehicles/2): list of pickup nodes
        deliveries (list size: num_vehicles/2): list of delivery nodes
        n_vehicle (int): number of vehicles
        weighted (bool): if True, demands will be generated in weights, else number of packages
        weight_range (tuple of ints): the range of the generated weight

    Returns:
        demands (ndarray size: num_cus + num_vehicles): the vehicle nodes are filled with 0, while pickup nodes are generated to be positive
            , and delivery nodes are equally negative
    '''
    demands = np.zeros(n_vehicle + 2*len(pickups)) 
    if weighted:
        pick_demand = np.array(np.random.choice(np.arange(weight_range[0], weight_range[1]), len(pickups)))
    else:
        pick_demand = np.ones(len(pickups))
    deli_demand = -pick_demand
    for i in range(len(pickups)):
        demands[pickups[i]] = pick_demand[i]
    for i in range(len(deliveries)):
        demands[deliveries[i]] = deli_demand[i]

    return demands
    
# def pickup_deli(orders):
#     pickups = []
#     deliveries = []
#     for request in orders:
#         pickups.append(request[0])
#         deliveries.append(request[1])
        
#     return pickups, deliveries

def create_data_model(n_cus=20, n_vehicle=1, scale=100, open=True, capacity=float('inf'), max_dist=float('inf')):
    '''
    create data model in the form of dict 

    Parameters:
        n_cus (int): number of customers (even number)
        n_vehicles (int): number of available vehicles
        scale (int/float): scale of (map) coordinates
        open (bool): open VRP, that is, open: does not return to depot, close: return to depot
    '''
    data = {}
    distance_mat, pickups_delivery_list, coords, pickups, deliveries = generate_random_samples(n_cus, n_vehicle, scale)
    data['coordinates'] = coords
    data['distance_matrix'] = distance_mat
    data['pickups_deliveries'] = pickups_delivery_list
    data['pickups'] = pickups
    data['deliveries'] = deliveries
    data['num_vehicles'] = n_vehicle
    data['depot'] = np.arange(n_vehicle)
    data['num_customers'] = n_cus
    data['starts'] = [*range(n_vehicle)]
    # data['ends'] = [n_cus + n_vehicle]*n_vehicle
    data['ends'] = [*range(n_vehicle)]
    data['demands'] = generate_demands(pickups, deliveries, n_vehicle)
    data['vehicle_capacities'] = np.array([capacity]*n_vehicle)    
    data['max_distance'] = np.array([max_dist]*n_vehicle)
    if open: 
        data['distance_matrix'][:,:n_vehicle] = 0

    return data
