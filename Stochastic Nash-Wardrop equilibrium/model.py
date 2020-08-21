# model parameters:
import copy
import numpy as np
import transport_graph as tg

import oracles
import dual_func_calculator as dfc

import universal_similar_triangles_method as ustm
import universal_gradient_descent_method as ugd

#from numba import jit
import math


class Model:
    def __init__(self, graph_data, graph_correspondences, total_od_flow, mu = 0.25, rho = 0.15, gamma = 1.):
        self.total_od_flow = total_od_flow
        self.mu = mu
        self.rho = rho
        self.gamma = gamma
        self.inds_to_nodes, self.graph_correspondences, graph_table = self._index_nodes(graph_data['graph_table'],
                                                                                        graph_correspondences)
        self.graph = tg.TransportGraph(graph_table, len(self.inds_to_nodes), graph_data['links number'])
        
    def _index_nodes(self, graph_table, graph_correspondences):
        table = graph_table.copy()
        inits = np.unique(table['init_node'][table['init_node_thru'] == False])
        terms = np.unique(table['term_node'][table['term_node_thru'] == False])
        through_nodes = np.unique([table['init_node'][table['init_node_thru'] == True], 
                                   table['term_node'][table['term_node_thru'] == True]])
        
        nodes = np.concatenate((inits, through_nodes, terms))
        nodes_inds = list(zip(nodes, np.arange(len(nodes))))
        init_to_ind = dict(nodes_inds[ : len(inits) + len(through_nodes)])
        term_to_ind = dict(nodes_inds[len(inits) : ])
        
        table['init_node'] = table['init_node'].map(init_to_ind)
        table['term_node'] = table['term_node'].map(term_to_ind)
        correspondences = {}
        for origin, dests in graph_correspondences.items():
            dests = copy.deepcopy(dests)
            correspondences[init_to_ind[origin]] = {'targets' : list(map(term_to_ind.get , dests['targets'])), 
                                                                     'corrs' : dests['corrs']}
            
        inds_to_nodes = dict(zip(range(len(nodes)), nodes))
        return inds_to_nodes, correspondences, table

    
    def find_equilibrium(self, solver_name = 'ustm', composite = True, solver_kwargs = {}):
        if solver_name == 'ustm':
            solver_func = ustm.universal_similar_triangles_method
            starting_msg = 'Universal similar triangles method...'
            if not 'L_init' in solver_kwargs:
                solver_kwargs['L_init'] = 0.1 * self.graph.max_path_length * self.total_od_flow / self.gamma
        elif solver_name == 'ugd':
            solver_func = ugd.universal_gradient_descent_method
            starting_msg = 'Universal gradient descent method...'
            if not 'L_init' in solver_kwargs:
                solver_kwargs['L_init'] = 0.1 * self.graph.max_path_length * self.total_od_flow / self.gamma
        else:
            raise NotImplementedError('Unknown solver!')
        
        phi_big_oracle = oracles.PhiBigOracle(self.graph, self.graph_correspondences, gamma = self.gamma)
        h_oracle = oracles.HOracle(self.graph.free_flow_times, self.graph.capacities, 
                                   rho = self.rho, mu = self.mu)
        primal_dual_calculator = dfc.PrimalDualCalculator(phi_big_oracle, h_oracle,
                                                          self.graph.free_flow_times, self.graph.capacities,
                                                          rho = self.rho, mu = self.mu)
        if composite == True:
            print('Composite optimization...')
            oracle = phi_big_oracle  
            prox = h_oracle.prox
        else:
            print('Non-composite optimization...')
            oracle = phi_big_oracle + h_oracle
            def prox_func(grad, point, A):
                """
                Computes argmin_{t: t \in Q} <g, t> + A / 2 * ||t - p||^2
                    where Q - the feasible set {t: t >= free_flow_times},
                    A - constant, g - (sub)gradient vector, p - point at which prox is calculated
                """
                return np.maximum(point - grad / A, self.graph.free_flow_times)
            prox = prox_func
        print('Oracles created...')
        print(starting_msg)
        
        result = solver_func(oracle, prox,
                             primal_dual_calculator, 
                             t_start = self.graph.free_flow_times,
                             **solver_kwargs)
        #TODO: add equilibrium travel times between zones
        return result
