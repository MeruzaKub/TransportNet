# from scipy.special import expit
#import multiprocessing as mp
from collections import defaultdict
#from scipy.misc import logsumexp
from scipy.special import expit
import numpy as np
import time
from transport_graph import JitTransportGraph
from numba.experimental import jitclass
from numba import jit, int32, int64, float64
from numba import njit

#TODO PhiBigOracle should memorize func and grad at least for 2 points.

@jit(["float64(float64[:])"])
def logsumexp(ns):
    nmax = np.max(ns)
    if np.isinf(nmax):
        return - np.inf
    ds = ns - nmax
    exp_sum = np.exp(ds).sum()
    return nmax + np.log(exp_sum)
  
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
        
    def prox(self, p, A, x_start = None):
        """
            Calculates prox of the function f(x) at point p:
            prox_f (p) = argmin_{x in Q} 0.5 ||x - p||^2 + A * f(x)
            p - point
            Q - feasible set
            A - constant
            x_start - start point for iterative minimization method 
        """
        raise NotImplementedError('Prox of the function is not implemented.')
        
    def __add__(self, other):
        return AdditiveOracle(self, other)

        
class AdditiveOracle(BaseOracle):
    def __init__(self, *oracles):
        self.oracles = oracles
                 
    def func(self, x):
        func = 0
        for oracle in self.oracles:
            func += oracle.func(x)
        return func
        
    def grad(self, x):
        grad = np.zeros(len(x))
        for oracle in self.oracles:
            grad += oracle.grad(x)
        return grad
    
    @property
    def time(self): #getter
        return np.sum([oracle.time for oracle in self.oracles])
        

class AutomaticOracle(BaseOracle):
    def __init__(self, source, graph, source_correspondences, gamma = 1.0):
        #stock graph
        self._graph = graph
        self._source = source
        corr_targets = np.array(graph.get_nodes_indices(source_correspondences['targets']), dtype = 'int64')
        corr_values = np.array(source_correspondences['corrs'], dtype = 'float64')
        nonzero_indices = np.nonzero(corr_values)
        corr_targets = corr_targets[nonzero_indices]
        corr_values = corr_values[nonzero_indices]
        
        self._jit_oracle = JitAutomaticOracle(graph.jit_graph, graph.get_node_index(source), 
                                              corr_targets, corr_values, gamma)
   
    def func(self, t_parameter):
        return self._jit_oracle.func(t_parameter)

    def grad(self, t_parameter):
        return self._jit_oracle.grad(t_parameter)


spec = [
    ('graph', JitTransportGraph.class_type.instance_type),
    ('nodes_number', int64),
    ('edges_number', int64),
    ('path_max_length', int64),
    ('source', int64),
    ('t_current', float64[:]),
    ('corr_targets', int64[:]),
    ('corr_values', float64[:]),
    ('targets', int64[:]),
    ('gamma', float64),
    ('A_values', float64[:,:]),
    ('B_values', float64[:,:]),
]

@jitclass(spec)
class JitAutomaticOracle:
    """
    Oracle for automatic calculations of function kGamma * \Psi (t)
    """

    def __init__(self, graph, source, corr_targets, corr_values, gamma):
        #stock graph
        self.graph = graph
        self.nodes_number = graph.nodes_number
        self.edges_number = graph.links_number
        self.path_max_length = graph.max_path_length
        
        self.source = source
        self.corr_targets = corr_targets
        self.corr_values = corr_values
        self.targets = np.zeros(self.edges_number, dtype = int64)
        for edge in range(self.edges_number):
            self.targets[edge] = graph.target_of_edge(edge)
        self.gamma = gamma
        
        self.t_current = np.zeros(self.edges_number)
        self.A_values = np.empty((self.path_max_length + 1, self.nodes_number))
        self.B_values = np.empty((self.path_max_length + 1, self.nodes_number))

        
    def func(self, t_parameter):
        #print('automatic func called...')
        self.t_current[:] = t_parameter
        self._calculate_a_b_values()
        
        return np.dot(self.corr_values,
                      self.B_values[self.path_max_length][self.corr_targets])

    
    def grad(self, t_parameter):
        #assert(np.all(self.t_current == t_parameter))
        #print('automatic grad called...')
        gradient_vector = np.zeros(self.edges_number)
        
        #psi_d_beta_values initial values path_length = kMaxPathLength 
        psi_d_beta_values = np.zeros(self.nodes_number)
        psi_d_beta_values[self.corr_targets] = self.corr_values
        psi_d_alpha_values = np.zeros(self.nodes_number)
        alpha_d_time = self._alpha_d_time_function(self.path_max_length)
        
        for path_length in range(self.path_max_length - 1, 0, -1):
            beta_d_beta_values = self._beta_d_beta_function(path_length + 1)
            psi_d_beta_values[:] = psi_d_beta_values * beta_d_beta_values
            
            #calculating psi_d_alpha_values
            beta_d_alpha_values = self._beta_d_alpha_function(path_length)
            
            psi_d_alpha_values = psi_d_beta_values * beta_d_alpha_values - \
                                 np.array([np.dot(psi_d_alpha_values[self.graph.successors(node)],
                                           alpha_d_time[self.graph.out_edges(node)]) for
                                           node in range(self.nodes_number)])
            
            #calculating gradient
            alpha_d_time = self._alpha_d_time_function(path_length)
            gradient_vector += psi_d_alpha_values[self.targets] * alpha_d_time
        #print('my result = ' + str(gradient_vector))
        return gradient_vector
    
        
    def _alpha_d_time_function(self, path_length):
        #print('alpha_d_time_func called...')
        result = np.zeros(self.edges_number)
        if path_length == 1:
            result[self.graph.out_edges(self.source)] = - 1.0
        else:
            for node in range(self.nodes_number):
                A_node = self.A_values[path_length][node]
                if not np.isinf(A_node):
                    A_source = self.A_values[path_length - 1][self.graph.predecessors(node)]
                    in_edges = self.graph.in_edges(node)
                    result[in_edges] = - np.exp((A_source - self.t_current[in_edges] - A_node) /
                                                self.gamma)
        return result
    
    
    def _beta_d_beta_function(self, path_length):
        if path_length == 1:
            return np.zeros(self.nodes_number)
        beta_new = self.B_values[path_length][:]
        beta_old = self.B_values[path_length - 1][:]
        
        indices = np.nonzero(np.logical_not(np.isinf(beta_new)))
        result = np.zeros(self.nodes_number)
        result[indices] = np.exp((beta_old[indices] - beta_new[indices]) / self.gamma)
        return result


    def _beta_d_alpha_function(self, path_length):
        if path_length == 1:
            return np.ones(self.nodes_number)
        alpha_values = self.A_values[path_length][:]
        beta_values = self.B_values[path_length][:]

        indices = np.nonzero(np.logical_not(np.isinf(beta_values)))
        result = np.zeros(self.nodes_number)
        result[indices] = np.exp((alpha_values[indices] - beta_values[indices]) / self.gamma)
        return result


    def _calculate_a_b_values(self):
        self.A_values = np.full(self.A_values.shape, - np.inf)
        self.B_values = np.full(self.B_values.shape, - np.inf)
        initial_values = - 1.0 * self.t_current[self.graph.out_edges(self.source)]
        self.A_values[1][self.graph.successors(self.source)] = initial_values
        self.B_values[1][self.graph.successors(self.source)] = initial_values
        
        for path_length in range(2, self.path_max_length + 1):
            for term_vertex in range(self.nodes_number):
                if len(self.graph.predecessors(term_vertex)) > 0:
                    alpha = self.gamma * logsumexp(1.0 / self.gamma * 
                            (self.A_values[path_length - 1][self.graph.predecessors(term_vertex)]
                            - self.t_current[self.graph.in_edges(term_vertex)]))
                    
                    beta = self.gamma * logsumexp(np.array([1.0 / self.gamma * 
                                                   self.B_values[path_length - 1][term_vertex],
                                                   1.0 / self.gamma * alpha]))
                    
                    self.A_values[path_length][term_vertex] = alpha
                    self.B_values[path_length][term_vertex] = beta


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
        self.entropy_current = None
        
        self.auto_oracles = []
        for source, source_correspondences in self.correspondences.items():
            self.auto_oracles.append(AutomaticOracle(source, self.graph, source_correspondences, gamma = self.gamma))
            
        self.time = 0.0
    
    def _reset(self, t_parameter):
        #print('Start reset')
        tic = time.time()
        self.t_current = t_parameter
        self.func_current = 0.0
        self.grad_current = np.zeros(self.graph.links_number)
        for auto_oracle in self.auto_oracles:
            self.func_current += auto_oracle.func(self.t_current)
            self.grad_current += auto_oracle.grad(self.t_current)
        self.entropy_current = self.func_current - np.dot(self.t_current, self.grad_current)
        self.time += time.time() - tic
        #print('Stop reset')
    
    def func(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            self._reset(t_parameter)
        return self.func_current
            
    def grad(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            self._reset(t_parameter)
        return self.grad_current
    
    def entropy(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            self._reset(t_parameter)
        return self.entropy_current

    
#Newton's method for HOracle
@njit
def newton(x_0_arr, a_arr, mu,
           tol = 1e-7, max_iter = 1000):
    """
    Newton method for equation x - x_0 + a x^mu = 0, x >= 0
    """
    res = np.empty(len(x_0_arr), dtype = np.float_)
    for i in range(len(x_0_arr)):
        x_0 = x_0_arr[i]
        a = a_arr[i]
        if x_0 <= 0:
            res[i] = 0
            continue
        x = min(x_0, (x_0 / a) ** (1 / mu))
        for it in range(max_iter):
            x_next = x - f(x, x_0, a, mu) / der_f(x, x_0, a, mu)
            if x_next <= 0:
                x_next = 0.1 * x
            x = x_next
            if np.abs(f(x, x_0, a, mu)) < tol:
                break
        res[i] = x
    return res

@njit
def f(x, x_0, a, mu):
    return x - x_0 + a * x ** mu

@njit
def der_f(x, x_0, a, mu):
    return 1.0 + a * mu * x ** (mu - 1)

class HOracle(BaseOracle):
    def __init__(self, freeflowtimes, capacities, rho = 10.0, mu = 0.25):  
        self.links_number = len(freeflowtimes)
        self.rho = rho
        self.mu = mu
        self.freeflowtimes = np.copy(freeflowtimes)
        self.capacities = np.copy(capacities)
        
        self.time = 0
    
    def func(self, t_parameter): 
        """
        Computes value of the function h(times) = \sum_i sigma^*(times[i])
        """
        if self.mu == 0:
            h_func = np.dot(self.capacities, np.maximum(t_parameter - self.freeflowtimes,0))
        else:
            h_func = np.sum(self.capacities * (t_parameter - self.freeflowtimes) * 
                                      (np.maximum(t_parameter - self.freeflowtimes, 0.0) / 
                                       (self.rho * self.freeflowtimes)) ** self.mu) / (1.0 + self.mu)
        return h_func
    
    def conjugate_func(self, flows):
        """
        Computes the conjugate of the function h(t):
        h^*(flows) = \sum_i sigma(flows[i]), since h(t) is a separable function
        """
        if self.mu == 0:
            return np.dot(self.freeflowtimes, flows) 
        else:
            return np.dot(self.freeflowtimes * flows, 
                          self.rho * self.mu / (1.0 + self.mu) * 
                          (flows / self.capacities) ** (1.0 / self.mu) + 1.0)
    
    def grad(self, t_parameter):
        if self.mu == 0:
            h_grad = self.capacities
        else:
            h_grad = self.capacities * (np.maximum(t_parameter - self.freeflowtimes, 0.0) / 
                                       (self.rho * self.freeflowtimes)) ** self.mu
        return h_grad
    
    def prox(self, grad, point, A):
        """
        Computes argmin_{t: t \in Q} <g, t> + A / 2 * ||t - p||^2 + h(t)
        where Q - the feasible set {t: t >= free_flow_times},
              A - constant, g - (sub)gradient vector, p - point at which prox is calculated
        """
        #rewrite as A/2 ||t - p_new||^2 + h(t)
        point_new = point - grad / A
        if self.mu == 0:
            return np.maximum(point_new - self.capacities / A, self.freeflowtimes)
        elif self.mu == 1:
            pass
        elif self.mu == 0.5:
            pass
        elif self.mu == 0.25:
            pass
        #rewrite as x - x_0 + a x^mu = 0, x >= 0
        #where x = (t - bar{t})/(bar{t} * rho), x_0 = (p_new - bar{t})/(bar{t} * rho),
        #      a = bar{f} / (A * bar{t} * rho)
        x = newton(x_0_arr = (point_new - self.freeflowtimes) / (self.rho * self.freeflowtimes),
                   a_arr = self.capacities / (A * self.rho * self.freeflowtimes),
                   mu = self.mu)
        argmin = (1 + self.rho * x) * self.freeflowtimes
        return argmin

"""    
    def pickle_func(oracle, args):
        return oracle._process_func(*args)

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

