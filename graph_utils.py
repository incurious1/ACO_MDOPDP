import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def create_graph_from_requests(pickups_deliveries): 
    '''
    create graph object from the pickups-deliveries requests (outdated) (not_used)
    '''
    pickups_DAG_list = []
    for request in pickups_deliveries:
        pickups_DAG_list.append((request[0],request[1]))
        pickups_DAG_list.append((0,request[0]))
    
    pickups_DAG = nx.DiGraph(pickups_DAG_list)

    return pickups_DAG
    
def draw_solution_path(path, n, data, n_vehicle = 1): 
    '''
    draw the solution (outdated)
    '''
    G = nx.Graph()
    pd_points = {}
    pd_points[0] = [0, 0]
    for i in range(len(data['pickups_deliveries'])):
        pd_points[data['pickups_deliveries'][i][0]] = ['P'+str(i+1), i+1]
        pd_points[data['pickups_deliveries'][i][1]] = ['D'+str(i+1), i+1]

    edges = [(pd_points[path[k]][0], pd_points[path[k+1]][0]) for k in range(n-1)]
    G.add_edges_from(edges)
    for i in range(n):
        G.add_node(pd_points[i][0], pos = tuple(data['coordinates'][i]))

    colors = []
    for k in G:
        if k == 0: 
            colors.append(0)
        else: 
            colors.append(int(k[1:]))
        
    nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, node_size=250, node_color=colors, 
            cmap = plt.cm.tab20)



def calculate_cost(path, distance_matrix):
    '''
    calculate the cost of a path from the distance matrix (including the cost from the last node back to the first node)

    Parameters:
        path (list of ints): input path (list of visited nodes)
        distance_matrix (ndarray): distance matrix for the calculation 

    Returns:
        cost (float): cost of the path
    '''
    return distance_matrix[np.roll(path, 1), path].sum()

def calculate_total_cost(solution, distance_matrix):
    '''
    calculate the cost of all vehicles' paths

    Parameters:
        solution (list size: num_vehicles): list of input paths
        distance_matrix (ndarray): distance matrix for the calculation 

    Returns:
        cost (ndarray size: num_vehicles): costs corresponding to each input solutions
    '''
    cost = np.zeros(len(solution))
    for i in range(len(solution)):
        cost[i] = calculate_cost(solution[i], distance_matrix)
    return cost

def weight_trace(path, data):
    '''
    track the weight of a vehicle at each node of the path it travels.

    Parameters:
        path (list of ints): input path (list of visited nodes)
        data (dict): data format

    Returns:
        carrying (list of floats): the weight carried by a vehicle at each node of the path
    '''
    carrying = []
    for k in path:
        carrying.append(carrying[-1] + data['demands'][k] if carrying else 0)
    return carrying

def total_weight_trace(solution, data):
    '''
    track the weights of all vehicles at each node of the path they travel.

    Parmeters:
        solution (list size: num_vehicles): list of input paths
        data (dict): data format

    Returns:
        total_carrying (list size: num_vehicles): list of all weight traces of all vehicles
    '''
    total_carrying = []
    for i in range(len(solution)):
        carrying = weight_trace(solution[i], data)
        total_carrying.append(carrying)
    return total_carrying

def valid_weight(path, data, capacity):
    '''
    check if the path satisfy capacity constaint, that is, all carrying weight in each node must not exceed the given capacity

    Parameters:
        path (list of ints): input path (list of visited nodes)
        data (dict): data format
        capacity (float): the capacity of the vehicle that travel through the input path

    Returns:
        (bool): whether the condition is satisfy or not
    '''
    return max(weight_trace(path, data)) <= capacity

def within_dist_limit(path, distance_matrix, max_dist):
    '''
    check if the path satisfy max distance constraint, that is, the total cost of each vehicle must not exceed the given max distance

    Parameters:
        path (list of ints): input path (list of visited nodes)
        distance_matrix (ndarray): distance matrix for the calculation 
        max_dist (float): maximum distance limit of the vehicle

    Returns:
        (bool): whether the condition is satisfy or not
    '''
    return calculate_cost(path, distance_matrix) <= max_dist

def check_request_order(solution, data):
    '''
    check if the given input solution satisfy the pickups-deliveries constaint

    Parameters:
        solution (list size: num_vehicles): list of input paths
        data (dict): data format

    Returns:
        (bool): whether the condition is satisfy or not
    '''
    pickups, deliveries = data['pickups'], data['deliveries']
    requests = {pickups[i]:deliveries[i] for i in range(len(pickups))}
    solution_temp = solution.copy()
    for subpath_ in solution_temp:
        subpath = subpath_.copy()
        depot = subpath.pop(0)
        for k in range(len(subpath)//2):
            node = subpath.pop(0)
            if (node not in pickups) or (requests[node] not in subpath): 
                return False
            else: 
                subpath.remove(requests[node])
        if subpath and (subpath[0] != depot): 
            return False     
    return True