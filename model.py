# model parameters:
import oracles
import dual_func_calculator as dfc

import phi_small_solver as pss
import universal_similar_triangles_function as ustf

import univ_grad_phi_small_solver as ug_pss
import universal_gradient_descent as ugd

import frank_wolfe_algorithm as fwa

from numba import jit
import math

@jit
def model_solve(graph, graph_correspondences, total_od_flow,
                solver_name = 'ustf',
                mu = 0.25, rho = 1.0, 
                epsilon = 1e-3, max_iter = 1000, verbose = False):
    if solver_name == 'fwa':
        solver_func = fwa.frank_wolfe_algorithm
        starting_msg = 'Frank-Wolfe algorithm...'
    elif solver_name == 'ustf':
        solver_func = ustf.universal_similar_triangles_function
        phi_small_solver_func = pss.PhiSmallSolver
        starting_msg = 'Universal similar triangles function...'
        L_init = 2.0 * math.sqrt(graph.max_path_length) * total_od_flow
    elif solver_name == 'ugd':
        solver_func = ugd.universal_gradient_descent_function
        phi_small_solver_func = ug_pss.UnivGradPhiSmallSolver
        starting_msg = 'Universal gradient descent...'
        L_init = 1.0
    else:
        print('Define function!')
    
    if solver_name == 'ustf' or solver_name == 'ugd':
        phi_small_solver = phi_small_solver_func(graph.freeflow_times, graph.capacities,
                                                 rho = rho, mu = mu)
    else:
        phi_small_solver = None
        L_init = None

    phi_big_oracle = oracles.PhiBigOracle(graph, graph_correspondences)
    primal_dual_calculator = dfc.PrimalDualCalculator(phi_big_oracle,
                                                      graph.freeflow_times, graph.capacities,
                                                      rho = rho, mu = mu)
    if verbose:
        print('Oracles created...')
        print(starting_msg)
    result = solver_func(phi_big_oracle, phi_small_solver,
                         primal_dual_calculator, 
                         graph.freeflow_times,
                         L_init, max_iter = max_iter, 
                         epsilon = epsilon, verbose = verbose)
    
    #t_average_w = phi_big_oracle.time_av_w_matrix(t_result)
    return result