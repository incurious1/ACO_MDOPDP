import graph_utils

# === 2-opt ==========================================================================================

def valid_segment(segment, data):
    for request in data['pickups_deliveries']:
        if set(request).issubset(set(segment)):
            return False
    else: return True

def two_opt_swap(path, v1, v2, data):
    if valid_segment(path[v2:v1:-1], data):
        new_path = path[:v1+1].copy()
        new_path.extend(path[v2:v1:-1])
        new_path.extend(path[v2+1:])
        if graph_utils.valid_weight(new_path, data, data['vehicle_capacities'][path[0]]):
            return new_path
    return []

def two_opt(cost, path, data):
    improvement = True
    while improvement: 
        improvement = False
        for i in range(1, len(path)-2):
            for j in range(i+2, len(path)):
                new_path = two_opt_swap(path,i,j,data)
                if len(new_path) == 0: continue
                new_cost = graph_utils.calculate_cost(new_path, data['distance_matrix'])
                if new_cost < cost:
                    path = new_path
                    cost = new_cost
                    improvement = True
    return cost, path

# ==================================================================================================

# === k-opt (Lin-Kernighan algorithm) ==============================================================

def k_opt(cost, S_init, data, k_max = 5):
    def around_node(node, subpath):
        index = subpath.index(node)
        pred = index - 1
        succ = index + 1

        if succ == len(subpath): succ = 0
        return (subpath[pred], subpath[succ])

    def find_segment(node, segment, current_seg):
            for segm in segment:
                if segm == current_seg: continue
                if segm[0] == node:
                    return segm, False, True
                elif segm[-1] == node:
                    return segm, True, True

            return [], True, False

    def generate_sol(subpath, p, data):
        broken = {(p[2*i],p[2*i+1]) for i in range(len(p)//2)}
        joined = {(p[2*i+1],p[2*i+2]) for i in range(len(p)//2-1)}
        joined.add((p[-1],p[0]))
        segment = []
        last_index = 0
        
        for k in range(len(subpath)-1):
            if ((subpath[k],subpath[k+1]) in broken) or ((subpath[k+1],subpath[k]) in broken):
                segment.append(subpath[last_index:k+1])
                last_index = k+1
        segment.append(subpath[last_index:])
     
        for k in range(len(segment)):
            for join_edge in joined:
                last_node = segment[0][-1]
                if join_edge[0] == last_node:
                    seg, rev, contain = find_segment(join_edge[1], segment, segment[0])
                elif join_edge[1] == last_node:
                    seg, rev, contain = find_segment(join_edge[0], segment, segment[0])
                else: continue

                if contain:
                    if rev:
                        segment[0] += seg[::-1]
                    else: segment[0] += seg
                        
                    segment.remove(seg)

        return (len(segment[0]) == len(subpath)) and graph_utils.check_request_order([segment[0]], data) \
                            and graph_utils.valid_weight(segment[0], data, data['vehicle_capacities'][subpath[0]]), segment[0]
    #     return (len(segment[0]) == len(subpath)), segment[0]

    def closest_neighbors(p2i, S_in, p, dm):
        closest = {}
        pk = p + [p2i]
        broken = {(pk[2*i],pk[2*i+1]) for i in range(len(pk)//2)}
        joined = {(pk[2*i+1],pk[2*i+2]) for i in range(len(pk)//2-1)}
        joined.add((pk[-1],p[0]))
        G = 0
        # for j in joined:
        #     G -= dm[j]
        # for j in broken:
        #     G += dm[j]
        G += dm[list(broken)].sum() - dm[list(joined)].sum()
        
        
        for i in range(len(S_in)):
            curr_node = S_in[i]
            if curr_node in pk:
                continue
                
            for succ in around_node(curr_node, S_in):
                if succ == pk[-1] and pk.count(succ) > 1:
                    continue              
                diff = G + dm[curr_node, succ] - dm[p2i, curr_node]
                if diff < 0 : continue
                if curr_node in closest and diff < closest[curr_node]:
                    continue
                else:  closest[curr_node] = diff
                    
        return sorted(closest.items(), key= lambda x: x[1], reverse=True)

    def condition_1(p2i, i, p):
        if i < 2: return True
        return not (p2i == p[-1] and p.count(p2i) > 1)

    def condition_2(p_temp, dm, i):
        diff = 0
        for m in range(i):
            diff += dm[p_temp[2*m-1],p_temp[2*m]] - dm[p_temp[2*m],p_temp[2*m+1]]
        if diff > 0: return True
        else: return False
        
    def kopt_main(S_in, p1, p, i, k_max, data):
        dm = data['distance_matrix']
        p_temp = p.copy()
        p_temp += [-1,-1]
        around = around_node(p[2*i-2], S_in)
        for p2i in around:
            p_temp[2*i-1] = p2i

            if not condition_1(p2i, i, p):
                continue
            if i >= 2 and (dm[p_temp[:2*i:2],p_temp[1:2*i:2]].sum() > \
                (dm[p_temp[1:2*i-1:2],p_temp[2:2*i-1:2]].sum() + dm[p_temp[2*i-1],p_temp[0]])) :
                
                is_path, S_out = generate_sol(S_in, p + [p2i], data)         
                if not is_path: continue
                return S_out, p

            if i == k_max: return (S_in, [])
            candidate_set = closest_neighbors(p2i, S_in, p, dm)
            for j in range(5):
                if j < len(candidate_set):
                    p2i1 = candidate_set[j][0]
                    p_temp[2*i] = p2i1
                    if not condition_2(p_temp, dm, i): 
                        continue
                    (S_temp, p_prime) = kopt_main(S_in, p1, p+[p2i,p2i1], i+1, k_max, data)
                    if not S_temp: continue
                    if graph_utils.calculate_cost(S_temp, dm) < graph_utils.calculate_cost(S_in, dm):
                        return (S_temp, p_prime)                     
        return (S_in, [])
  
    node_set = set(S_init)
    S_curr = S_init.copy()
    while node_set:
        p1 = node_set.pop()
        p = [p1]
        S_temp, p = kopt_main(S_curr, p1, p, 1, k_max, data)
        cost_temp = graph_utils.calculate_cost(S_temp, data['distance_matrix'])

        if cost_temp < cost and graph_utils.check_request_order([S_temp], data):
            S_curr = S_temp
            cost = cost_temp
            node_set = set(S_init)

    return cost, S_curr

# ==================================================================================================

def local_search(cost, path, data, method = None):
    '''
    implement local search algorithms e.g., 2-opt or LKH (k-opt).
    Parameters: 
        cost (ndarray): current cost (size: num_vehicle)
        path (list): current solution path (size: num_vehicle)
        data (dict): data format
        method ({'two_opt','k_opt'}): choose method
    Returns:  
        cost (ndarray): output cost after local search (size: num_vehicle)
        path (list): output solution path after local search (size: num_vehicle)
    '''
    if method is None: return cost, path 
    if method == 'two_opt':
        for i in range(data['num_vehicles']):
            local_cost, local_path = two_opt(cost[i], path[i], data)
            cost[i], path[i] = local_cost, local_path
    elif method == 'k_opt':
        for i in range(data['num_vehicles']):
            local_cost, local_path = k_opt(cost[i], path[i], data)
            cost[i], path[i] = local_cost, local_path
    else: raise KeyError('method does not exist.')
    
    return cost, path  