import numpy as np

def calculate_prob(parameters, path_quality_matrix, pheromone_matrix, possible_edges):
    '''
    calculate probability for each ant's next node in ant colony optimization.
    '''
    prob = pheromone_matrix[possible_edges.astype(bool)]**parameters['alpha'] \
        *path_quality_matrix[possible_edges.astype(bool)]**parameters['beta']
    prob /= prob.sum()
    return prob

def generate_solution(data, method, pheromone_matrix=None, path_quality_matrix=None, parameters=None):

    n = len(data['distance_matrix'])            # num nodes
    pickups, deliveries = data['pickups'], data['deliveries']
    num_vehicles = data['num_vehicles']
    num_customers = data['num_customers']
    capacity = data['vehicle_capacities']       # capacity
    current_node = np.array(data['starts'])     # starting point

    requests = {pickups[i]:deliveries[i] for i in range(len(pickups))}

    cost = np.zeros(num_vehicles)
    path = [[i] for i in range(num_vehicles)]
    carrying_weight = np.zeros(num_vehicles)

    vehicle_id = np.arange(num_vehicles)
    to_deliver = [np.array([], int) for _ in range(num_vehicles)]

    pickups_matrix = np.zeros((n,n), int)
    pickups_matrix[:, pickups] = 1

    vehicle_matrix = np.zeros((n,n), int)
    vehicle_matrix[current_node, :] = 1

    from_node = np.outer(np.arange(n), np.ones(n, int))
    to_node = np.outer(np.ones(n, int), np.arange(n))

    visited_matrix = np.ones((n,n), int)
    visited_matrix[:, current_node] = 0

    for _ in range(num_customers):
        edges = np.ones((n,n), bool)

        if np.isfinite(capacity).all():
            available = np.zeros((n,n), int)
            available[current_node] = np.outer(capacity - carrying_weight, np.ones(n, int))
            mask = (np.outer(np.ones(n, int), data['demands']) <= available)
        else: mask = np.ones((n,n), int)

        possible_edges = edges * mask * pickups_matrix * vehicle_matrix * visited_matrix
        np.fill_diagonal(possible_edges, 0)

        if method == 'greedy':
            edge_index = data['distance_matrix'][possible_edges.astype(bool)].argmin()
        elif method == 'aco':
            prob = calculate_prob(parameters, path_quality_matrix, pheromone_matrix, possible_edges)
            edge_index =  np.random.choice(np.count_nonzero(possible_edges), 1, p = prob)[0]
        elif method == 'random':
            prob = np.ones(np.count_nonzero(possible_edges))/np.count_nonzero(possible_edges)
            edge_index =  np.random.choice(np.count_nonzero(possible_edges), 1, p = prob)[0]
        else: raise ValueError('method does not exist.')

        selected_edge = [from_node[possible_edges.astype(bool)][edge_index] \
            , to_node[possible_edges.astype(bool)][edge_index]]

        cost[vehicle_id[current_node == selected_edge[0]]] += \
            data['distance_matrix'][selected_edge[0],selected_edge[1]]
        path[vehicle_id[current_node == selected_edge[0]][0]].append(selected_edge[1])
        carrying_weight[vehicle_id[current_node == selected_edge[0]]] += data['demands'][selected_edge[1]]

        vehicle_matrix[selected_edge[0]] = 0
        vehicle_matrix[selected_edge[1]] = 1

        pickups_matrix[:, selected_edge[1]] = 0
        if selected_edge[1] in pickups:
            to_deliver[vehicle_id[current_node == selected_edge[0]][0]] = \
                np.append(to_deliver[vehicle_id[current_node == selected_edge[0]][0]], requests[selected_edge[1]])

        pickups_matrix[selected_edge[1], to_deliver[vehicle_id[current_node == selected_edge[0]][0]]] = 1

        current_node = np.where(current_node == selected_edge[0], selected_edge[1], current_node)
        visited_matrix[:, selected_edge[1]] = 0

    total_cost = cost.sum()
    
    return total_cost, cost, path

def nearest_heuristic(data):
    return generate_solution(data, 'greedy')

def generate_ants_path(data, pheromone_matrix, path_quality_matrix, parameters):
    return generate_solution(data, 'aco', pheromone_matrix, path_quality_matrix, parameters)