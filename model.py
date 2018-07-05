# model parameters:
import oracles
import dual_func_calculator as dfc

import phi_small_solver as pss
import universal_similar_triangles_corrected_to_check as ustf
#import universal_similar_triangles_function as ustf

import univ_grad_phi_small_solver as ug_pss
import universal_gradient_descent as ugd

from numba import jit
import math


@jit
def model_solve(graph, graph_correspondences, total_od_flow,
                solver_name = 'ustf',
                gamma = 1.0, mu = 0.25, rho = 1.0, 
                epsilon = 1e-3, max_iter = 1000, verbose = False):
    if solver_name == 'ustf':
        solver_func = ustf.universal_similar_triangles_function
        phi_small_solver_func = pss.PhiSmallSolver
        starting_msg = 'Universal similar triangles function...'
    elif solver_name == 'ugd':
        solver_func = ugd.universal_gradient_descent_function
        phi_small_solver_func = ug_pss.UnivGradPhiSmallSolver
        starting_msg = 'Universal gradient descent...'
    else:
        print('Define function!')
    
    #L_init = 1.0
    L_init = 0.1 * graph.max_path_length * total_od_flow / gamma
    phi_small_solver = phi_small_solver_func(graph.free_flow_times, graph.capacities,
                                             rho = rho, mu = mu)

    phi_big_oracle = oracles.PhiBigOracle(graph, graph_correspondences, gamma = gamma)
    primal_dual_calculator = dfc.PrimalDualCalculator(phi_big_oracle,
                                                      graph.free_flow_times, graph.capacities,
                                                      rho = rho, mu = mu)
    if verbose:
        print('Oracles created...')
        print(starting_msg)
    result = solver_func(phi_big_oracle, phi_small_solver,
                         primal_dual_calculator, 
                         graph.free_flow_times,
                         L_init, max_iter = max_iter, 
                         epsilon = epsilon, verbose = verbose)
    
    #t_average_w = phi_big_oracle.time_av_w_matrix(t_result)
    return result