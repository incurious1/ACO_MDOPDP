'''
initial solution method 
(1) nearest_heuristic
(2) insertion (less variance between each vehicles) (not finished)

'''
import generate_solution

def initial_solution(data, method):
    if method == 'nearest_heuristic':
        return generate_solution.nearest_heuristic(data)
