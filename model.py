# model parameters:
import oracles
import phi_small_solver as pss
import dual_func_calculator as dfc
import universal_similar_triangles_function as ustf

def model_solve(graph, graph_correspondences, total_od_flow, gamma = 1.0, mu = 0.25, rho = 1.0, 
                epsilon = 1e-5, max_iter = 1000, verbose = False): 
    phi_big_oracle = oracles.PhiBigOracle(graph = graph,
                                          correspondences = graph_correspondences, gamma = gamma)
    phi_small_solver = pss.PhiSmallSolver(phi_big_oracle, graph.freeflowtimes(), graph.capacities(),
                                          rho = rho, mu = mu)
    dual_func_calculator = dfc.DualFuncCalculator(phi_big_oracle,
                                                  graph.freeflowtimes(), graph.capacities(),
                                                  rho = rho, mu = mu)
    if verbose:
        print('Oracles created...')
        print('Universal similar triangles function...')
    L_init = 0.1 * graph.kMaxPathLength * total_od_flow / gamma
    t_result, iterations_number, res_msg = ustf.universal_similar_triangles_function(phi_big_oracle, phi_small_solver,
                                                                        dual_func_calculator, graph.freeflowtimes(),
                                                                        L_init, max_iter = max_iter, 
                                                                        epsilon = epsilon, verbose = verbose)
    flows = - phi_big_oracle.grad(t_result)
    return {'times': t_result, 'flows': flows, 'iter_num': iterations_number, 'res_msg': res_msg}