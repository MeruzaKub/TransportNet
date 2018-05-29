# model parameters:
import oracles
import phi_small_solver as pss
import dual_func_calculator as dfc
import universal_similar_triangles_function as ustf
from numba import jit

@jit
def model_solve(graph, graph_correspondences, total_od_flow, gamma = 1.0, mu = 0.25, rho = 1.0, 
                epsilon = 1e-3, max_iter = 1000, verbose = False): 
    phi_big_oracle = oracles.PhiBigOracle(graph, graph_correspondences, gamma = gamma)
    phi_small_solver = pss.PhiSmallSolver(graph.free_flow_times, graph.capacities,
                                          rho = rho, mu = mu)
    primal_dual_calculator = dfc.PrimalDualCalculator(phi_big_oracle,
                                                      graph.free_flow_times, graph.capacities,
                                                      rho = rho, mu = mu)
    if verbose:
        print('Oracles created...')
        print('Universal similar triangles function...')
    L_init = 0.1 * graph.max_path_length * total_od_flow / gamma
    result = ustf.universal_similar_triangles_function(phi_big_oracle, phi_small_solver,
                                                       primal_dual_calculator, 
                                                       graph.free_flow_times,
                                                       L_init, max_iter = max_iter, 
                                                       epsilon = epsilon, verbose = verbose)
    
    
    #t_average_w = phi_big_oracle.time_av_w_matrix(t_result)
    return result