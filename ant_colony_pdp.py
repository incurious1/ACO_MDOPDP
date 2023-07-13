import graph_utils
import numpy as np
import generate_solution
import local_search
import initial_solution

def generate_ant_paths(data, pheromone_matrix, path_quality_matrix, parameters):
    return generate_solution.generate_ants_path(data, pheromone_matrix, path_quality_matrix, parameters)

def initialize_parameters(alpha=2, beta=3, rho=0.8, sigma=None, gamma=0, zeta=4, penalty=2):
    parameters = {}
    parameters['alpha'] = alpha     # pheromone
    parameters['beta'] = beta       # path quality
    parameters['rho'] = rho         # phero evaporation coeff
    if sigma is None:               # phero update coeff
        parameters['sigma'] = 1-rho
    else: 
        parameters['sigma'] = sigma
    parameters['gamma'] = gamma     # weight coeff
    parameters['zeta'] = zeta       # local search depth
    parameters['penalty'] = penalty # penalty for max dist
    return parameters

def calculate_prob(parameters, path_quality_matrix, pheromone_matrix, possible_edges):
    prob = pheromone_matrix[possible_edges.astype(bool)]**parameters['alpha'] \
        *path_quality_matrix[possible_edges.astype(bool)]**parameters['beta']
    prob /= prob.sum()
    return prob

def update_pheromone(n, parameters, pheromone_matrix, ant_solutions, best_known, method='default'):
    change = np.zeros((n,n))
    if method in ['default','ants_and_best']:
        for k in range(len(ant_solutions)):
            for path in ant_solutions[k][2]:
                # change[np.roll(path, 1)[1:], path[1:]] += (parameters['zeta'] - k)*best_known[0] \
                #     / (ant_solutions[k][0]*parameters['zeta'])
                change[np.roll(path, 1)[1:], path[1:]] += (parameters['zeta'] - k)/(ant_solutions[k][0])

        for path in best_known[2]:
            # change[np.roll(path, 1)[1:], path[1:]] += ant_solutions[0][0] / best_known[0]
            change[np.roll(path, 1)[1:], path[1:]] += 1 / best_known[0]

        # pheromone_matrix = pheromone_matrix*parameters['rho'] + change*parameters['sigma']
        pheromone_matrix = pheromone_matrix*parameters['rho'] + change

    elif method == 'ants':
        for k in range(len(ant_solutions)):
            for path in ant_solutions[k][2]:             
                # change[np.roll(path, 1)[1:], path[1:]] += (parameters['zeta'] - k)*best_known[0] \
                #     / (ant_solutions[k][0]*parameters['zeta'])
                change[np.roll(path, 1)[1:], path[1:]] += (parameters['zeta'] - k)*best_known[0]/(ant_solutions[k][0]**2)

        pheromone_matrix = pheromone_matrix*parameters['rho'] + change
    return pheromone_matrix

def not_valid_dist(ant_path, data):
    for i in range(len(ant_path[1])):
        if ant_path[1][i] > data['max_distance'][i]:
            return True
    return False

def add_penalty_cost(cost, itr, data, parameters):
    new_cost = cost + itr * parameters['penalty'] * (cost - data['max_distance'])[cost - data['max_distance'] >= 0].sum()
    return new_cost.sum()

def ant_colony_pdp(data, num_ants=30, max_generation=100, ls_method='two_opt', max_penalty=True, init_method='nearest_heuristic' \
                  , stopping=20, load_balance_acc=False, init_pheromone=None, parameters=None):
    '''
    ant colony optimization algorithm for paired pickups and deliveries problem

    Parameters:
        data: dict
        num_ants (int): number of ants in one generation
        generation (int): number of generations (iterations)
        ls_method (str): local search method ['two_opt','k_opt']
        max_penalty (bool): use penalty method to satisfy max distance constraints
        stopping (int): number of same best solution for termination condition

    Returns:
        total_cost (float): sum of all vehicle's cost
        cost (ndarray): cost of each vehicle (size: num_vehicles)
        path (list): solution path (size: num_vehicles) 
    '''
    n_nodes = len(data['distance_matrix'])
    
    try:
        _ = init_pheromone.shape
        pheromone_matrix = init_pheromone
    except AttributeError:
        pheromone_matrix = np.ones((n_nodes,n_nodes))
        
    # if init_pheromone.any() != None:
    #     pheromone_matrix = np.ones((n_nodes,n_nodes))
    # else:
    #     pheromone_matrix = init_pheromone
    path_quality_matrix = data['distance_matrix'].copy()
    path_quality_matrix[path_quality_matrix > 0] = 1 / path_quality_matrix[path_quality_matrix > 0]
    
    if parameters is None:
        parameters = initialize_parameters()

    ## initial solution (might not satisfy max distance condition yet)
    best_known = list(initial_solution.initial_solution(data, init_method))
    if ls_method in ['two_opt','k_opt']:
        best_known[1], best_known[2] = local_search.local_search(best_known[1], best_known[2], data, method=ls_method)
    elif ls_method != 'None':
        raise KeyError('method does not exist.')
    best_known[0] = best_known[1].sum()

    count = 0
    for itr in range(max_generation):
        ant_solutions = []  
        ## generate ant solutions
        for _ in range(num_ants):
            ant_path = generate_ant_paths(data, pheromone_matrix, path_quality_matrix, parameters)
            ant_solutions.append(list(ant_path))

        ## filter out worse solutions to lower local search time (keep best zeta solutions) 
        ant_solutions.sort(key=lambda x: x[0]) ###
        ant_solutions = ant_solutions[:parameters['zeta']] ###
        
        ## use local search method in
        if ls_method in ['two_opt','k_opt']:                 
            solutions_done = {}
            for solution in ant_solutions:
                if solution[0] not in solutions_done.keys():
                    solution_cost = solution[0]
                    solution[1], solution[2] = local_search.local_search(solution[1], solution[2], data, method=ls_method)
                    solution[0] = solution[1].sum()
                    solutions_done[solution_cost] = solution[2].copy()
                else:
                    solution[2] = solutions_done[solution[0]].copy()
                    solution[1] = graph_utils.calculate_total_cost(solution[2], data['distance_matrix'])
                if max_penalty:
                    solution[0] = add_penalty_cost(solution[1], itr, data, parameters)
                if load_balance_acc and (itr < 20 or count < 5):
                # if load_balance_acc:
                    solution[0] += (max(solution[1]) - min(solution[1]))
                    
        ant_solutions.sort(key=lambda x: x[0])

        ## do local search for every ants 
        # ant_solutions = ant_solutions[:parameters['zeta']] ###

        if max_penalty:
            best_known[0] = add_penalty_cost(best_known[1], itr, data, parameters)
        ## if better solution is found, update.
        
        if load_balance_acc and (itr < 20 or count < 5):
        # if load_balance_acc:
            best_known[0] = best_known[1].sum() + max(solution[1]) - min(solution[1])
            
        if ant_solutions[0][0] < best_known[0]:
            best_known = ant_solutions[0]
            count = 0
        else: count += 1
        ## early termination condition => if no better solution found for 20 generations
        if count >= stopping:
            break
        ## update pheromone matrix
        pheromone_matrix = update_pheromone(n_nodes, parameters, pheromone_matrix, ant_solutions, best_known)

    best_known[0] = best_known[1].sum()
    return best_known, itr
