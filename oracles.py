# from scipy.special import expit
import multiprocessing as mp
from collections import defaultdict
from scipy.misc import logsumexp
from scipy.special import expit
import numpy as np


class BaseOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class AutomaticOracle(BaseOracle):
    """
    Oracle for automatic calculations of function kGamma * \Psi (t)
    """

    def __init__(self, source, graph, source_correspondences, gamma = 1.0):
    #may be it should be self.graph.kMaxPathLength
        self.graph = graph #should it be here? 
        self.source = source
        self.source_correspondences = source_correspondences
        self.gamma = gamma
        
        self.t_current = None

    def func(self, t_parameter):
        #print('automatic func called...'+'t_parameter = ' + str(t_parameter) )
        self.t_current = t_parameter
        self.calculate_a_b_values()
        return np.sum(self.source_correspondences.values() *
                      get_matrix_values(self.B_values, self.graph.kMaxPathLength,
                                        self.source_correspondences.keys()))

    def grad(self, t_parameter):
        assert(np.all(self.t_current == t_parameter))
        #print('automatic grad called...'+'t_parameter = ' + str(t_parameter) )
        gradient_vector = np.zeros(self.graph.kLinksNumber)
        
        #psi_d_beta_values initial values path_length = kMaxPathLength 
        psi_d_beta_values = np.zeros(self.graph.kNodesNumber)
        psi_d_beta_values[np.array(self.source_correspondences.keys()) - 1] = self.source_correspondences.values()
        psi_d_alpha_values = np.zeros(self.graph.kNodesNumber)
        
        for path_length in range(self.graph.kMaxPathLength - 1, 0, -1):
            beta_d_beta_values = self.beta_d_beta_function(path_length + 1)
            psi_d_beta_values = psi_d_beta_values * beta_d_beta_values
        
            #calculating psi_d_alpha_values
            beta_d_alpha_values = self.beta_d_alpha_function(path_length)
            first_terms = psi_d_beta_values * beta_d_alpha_values
                
            second_terms = np.zeros(self.graph.kNodesNumber)
            for node in self.graph.nodes():
                if len(self.graph.successors(node)) > 0:
                    alpha_d_alpha_values = np.array([self.alpha_d_alpha_functions(path_length + 1, term_vertex, node)
                                                     for term_vertex in self.graph.successors(node)])
                    second_term_value = np.sum(psi_d_alpha_values[np.array(self.graph.successors(node)) - 1] * \
                                               alpha_d_alpha_values)
                    #not good. Seems like you should know indexation of nodes
                    #np.vectorize(graph.nodes())
                    second_terms[node - 1] = second_term_value
            
            psi_d_alpha_values = first_terms + second_terms
        
            #calculating gradient
            for edge_index in self.graph.edges():
                target_vertex = self.graph.target_of_edge(edge_index)
                gradient_vector[edge_index] += psi_d_alpha_values[target_vertex - 1] * \
                                               self.alpha_d_time_function(path_length, target_vertex, edge_index)
        #print('my result = ' + str(gradient_vector))
        return gradient_vector
    
    def alpha_d_time_function(self, path_length, term_vertex, edge_index):
        #print('alpha_d_time_func called...')
        assert(not np.any(np.isnan(self.t_current)))
        """
        edges_indices = self.graph.in_edges(term_vertex)
        if edge_index not in edges_indices:
            return 0.0
        if path_length == 1:
            if self.graph.source_of_edge(edge_index) == self.source:
                return - 1.0
            else:
                return 0.0

        alpha_values = get_matrix_values(self.A_values, path_length - 1,
                                         self.graph.predecessors(term_vertex))
        assert(not np.any(np.isnan(alpha_values)))
        
        if np.all(np.isinf(alpha_values)) or np.isinf(alpha_values[edges_indices.index(edge_index)]):
            return 0.0
        values = alpha_values - self.t_current[edges_indices] - \
                 alpha_values[edges_indices.index(edge_index)] + self.t_current[edge_index]
        derivative_value = - 1.0 / np.sum(np.exp(values / self.gamma))
        assert(not np.any(np.isnan(derivative_value)))
        return derivative_value
        """
        if self.graph.target_of_edge(edge_index) != term_vertex:
            return 0.0
        edge_source = self.graph.source_of_edge(edge_index)
        if path_length == 1:
            if  edge_source == self.source:
                return - 1.0
            else:
                return 0.0
        A_value_term = get_matrix_values(self.A_values, path_length, term_vertex)
        A_value_source = get_matrix_values(self.A_values, path_length - 1, edge_source)
        if np.isinf(A_value_term) or np.isinf(A_value_source):
            return 0.0
        return - np.exp(1.0 / self.gamma * (A_value_source - 
                                            self.t_current[edge_index] - A_value_term))

        
    def alpha_d_alpha_functions(self, path_length, term_vertex, deriv_term_vertex):
        edge_index = self.graph.edge_index(deriv_term_vertex, term_vertex)
        result = - self.alpha_d_time_function(path_length, term_vertex, edge_index)
        assert(not np.any(np.isnan(result)))
        return result

    def beta_d_beta_function(self, path_length):
        #if path_length == 1:
        #    return np.zeros(self.graph.kNodesNumber)
        alpha_values = get_matrix_values(self.A_values, path_length)
        beta_values = get_matrix_values(self.B_values, path_length - 1)
        
        indices = np.nonzero(np.logical_not(np.isinf(beta_values)))
        values = - np.inf * np.ones(self.graph.kNodesNumber)
        values[indices] = - alpha_values[indices] + beta_values[indices]
        #values = np.where(np.logical_and(np.isinf(alpha_values), np.isinf(beta_values)), -np.inf, alpha_values - beta_values)
        result = expit(values / self.gamma)
        return result

    
    def beta_d_alpha_function(self, path_length):
        if path_length == 1:
            return np.ones(self.graph.kNodesNumber)
        alpha_values = get_matrix_values(self.A_values, path_length)
        beta_values = get_matrix_values(self.B_values, path_length - 1)
        
        indices = np.nonzero(np.logical_not(np.isinf(alpha_values)))
        values = - np.inf * np.ones(self.graph.kNodesNumber)
        values[indices] = alpha_values[indices] - beta_values[indices]
        #values = np.where(np.logical_and(np.isinf(alpha_values), np.isinf(beta_values)), -np.inf, alpha_values - beta_values)
        result = expit(values / self.gamma)
        return result

            
    def calculate_a_b_values(self):
        self.A_values = - np.inf * np.ones((self.graph.kMaxPathLength, self.graph.kNodesNumber))
        self.B_values = - np.inf * np.ones((self.graph.kMaxPathLength, self.graph.kNodesNumber))
        initial_values = - 1.0 * self.t_current[self.graph.out_edges(self.source)]
        set_matrix_values(initial_values, self.A_values, 1, self.graph.successors(self.source))
        set_matrix_values(initial_values, self.B_values, 1, self.graph.successors(self.source))
        
        for path_length in range(2, self.graph.kMaxPathLength + 1):
            for term_vertex in self.graph.nodes():
                if len(self.graph.predecessors(term_vertex)) > 0:
                    alpha = self.gamma * \
                            logsumexp(1.0 / self.gamma * 
                                      (get_matrix_values(self.A_values, path_length - 1, self.graph.predecessors(term_vertex))
                                       - self.t_current[self.graph.in_edges(term_vertex)]))

                    beta = self.gamma * \
                            logsumexp([1.0 / self.gamma * get_matrix_values(self.B_values, path_length - 1, term_vertex),
                                       1.0 / self.gamma * alpha])

                    set_matrix_values(alpha, self.A_values, path_length, term_vertex)
                    set_matrix_values(beta, self.B_values, path_length, term_vertex)
                    
                    #print('path_length = ' + str(path_length) + ' term_vertex = ' + str(term_vertex))
                    #print(get_matrix_values(self.A_values, path_length - 1, self.graph.predecessors(term_vertex)))
                    #print(self.t_current[self.graph.in_edges(term_vertex)])
                    #print('a_value = ' + str(alpha))
                    #print('b_value = ' + str(beta))
         
        assert(not np.any(np.isnan(self.A_values)))
        assert(not np.any(np.isnan(self.B_values)))
        
        
def set_matrix_values(values, array, path_length, vertices_list = None):
    if vertices_list:
        array[path_length - 1][np.array(vertices_list) - 1] = np.array(values)
    else:
        array[path_length - 1][:] = np.array(values)

def get_matrix_values(array, path_length, vertices_list = None):
    if vertices_list:
        res = array[path_length - 1][np.array(vertices_list) - 1]
    else:
        res = array[path_length - 1][:]
    return res

def pickle_func(oracle, args):
    return oracle._process_func(*args)

class PhiBigOracle(BaseOracle):
    def __init__(self, graph, correspondences, processes_number = None, gamma = 1.0):
        self.graph = graph
        self.correspondences = correspondences
        if processes_number:
            self.processes_number = processes_number
        else:
            self.processes_number = len(correspondences)
        self.gamma = gamma
        self.t_current = None
        self.func_current = None
        self.grad_current = None
        
        self.auto_oracles = []
        for source, source_correspondences in self.correspondences.iteritems():
            self.auto_oracles.append(AutomaticOracle(source, self.graph, source_correspondences, gamma = self.gamma))
            
    def _reset(self, t_parameter):
        self.t_current = t_parameter
        self.func_current = 0.0
        self.grad_current = np.zeros(self.graph.kLinksNumber)
        for auto_oracle in self.auto_oracles:
            self.func_current += auto_oracle.func(self.t_current)
            self.grad_current += auto_oracle.grad(self.t_current)
    
    def func(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            self._reset(t_parameter)
        return self.func_current
            
    def grad(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            self._reset(t_parameter)
        return self.grad_current
"""    
    def _process_func(self, source, graph,
                      source_correspondences, operation, t_parameter):
        automatic_oracle = AutomaticOracle(source, graph, source_correspondences)
        if operation == 'func':
            res = automatic_oracle.func(t_parameter)
        if operation == 'grad':
            res = automatic_oracle.grad(t_parameter)
        return res
                        
    def func(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            self.t_current = t_parameter
            #pool = mp.Pool(processes = self.processes_number)
            results = []
            for key, value in self.correspondences.iteritems():
                #results.append(pool.apply_async(pickle_func, args=(self, (key, self.graph, value, 'func', t_parameter))))
                results.append(pickle_func(self, (key, self.graph, value, 'func', t_parameter)))
            #results = np.array([p.get() for p in results])
            results = np.array(results)
            self.func_current = np.sum(results)
            pool.close()
        return self.func_current
    
    def grad(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            pool = mp.Pool(processes = self.processes_number)
            results = []
            for key, value in self.correspondences.iteritems():
                #results.append(pool.apply_async(pickle_func, args=(self, (key, self.graph, value, 'grad', t_parameter))))
                results.append(pickle_func(self, (key, self.graph, value, 'grad', t_parameter)))
            #results = np.array([p.get() for p in results])
            results = np.array(results)
            self.grad_current = np.sum(results, axis = 0)
            pool.close()
        return self.grad_current
"""
        

        