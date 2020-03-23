# model parameters:
import transport_graph as tg

import oracles
import dual_func_calculator as dfc
from prox_h import ProxH

import universal_similar_triangles_function as ustf

import universal_gradient_descent as ugd

import frank_wolfe_algorithm as fwa


class Model:
    def __init__(self, graph_data, graph_correspondences, total_od_flow, mu = 0.25, rho = 0.15):
        self.graph = tg.TransportGraph(graph_data)
        self.graph_correspondences = graph_correspondences
        self.total_od_flow = total_od_flow
        self.mu = mu
        self.rho = rho
        
    def find_equilibrium(self, solver_name = 'ustf', solver_kwargs = {}, verbose = False, save_history = False):
        if solver_name == 'fwa':
            solver_func = fwa.frank_wolfe_algorithm
            starting_msg = 'Frank-Wolfe algorithm...'
        elif solver_name == 'ustf':
            solver_func = ustf.universal_similar_triangles_function
            starting_msg = 'Universal similar triangles function...'
            if not 'L_init' in solver_kwargs:
                solver_kwargs['L_init'] = self.graph.max_path_length**0.5 * self.total_od_flow
        elif solver_name == 'ugd':
            solver_func = ugd.universal_gradient_descent_function
            starting_msg = 'Universal gradient descent...'
            if not 'L_init' in solver_kwargs:
                solver_kwargs['L_init'] = 1.0
        elif solver_name == 'sd':
            solver_func = ugd.universal_gradient_descent_function
            starting_msg = 'Subgradient descent...'
        else:
            raise NotImplementedError('Unknown solver!')

        if solver_name in ['ugd', 'ustf', 'sd']:
            prox_h = ProxH(self.graph.freeflow_times, self.graph.capacities, mu = self.mu, rho = self.rho)


        phi_big_oracle = oracles.PhiBigOracle(self.graph, self.graph_correspondences)
        primal_dual_calculator = dfc.PrimalDualCalculator(phi_big_oracle,
                                                          self.graph.freeflow_times, self.graph.capacities,
                                                          mu = self.mu, rho = self.rho)
        if verbose:
            print('Oracles created...')
            print(starting_msg)
        
        if solver_name == 'fwa':
            result = solver_func(phi_big_oracle,
                                 primal_dual_calculator, 
                                 t_start = self.graph.freeflow_times,
                                 verbose = verbose, save_history = save_history, **solver_kwargs)
        else:
            result = solver_func(phi_big_oracle, prox_h,
                                 primal_dual_calculator, 
                                 t_start = self.graph.freeflow_times,
                                 verbose = verbose, save_history = save_history, **solver_kwargs)

        return result