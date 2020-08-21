#import multiprocessing as mp
from collections import defaultdict
#from scipy.misc import logsumexp
import numpy as np
import time
import numba
from numba import njit
from numba.typed import List, Dict


@njit
def get_tree_order(nodes_number, targets, pred_arr):
    #get nodes visiting order for flow calculation
    visited = np.zeros(nodes_number, dtype = np.bool_)
    sorted_vertices = List()
    sorted_vertices.append(targets[0])
    sorted_vertices.clear()
    for vertex in targets:
        temp = List()
        while (not visited[vertex]):
            visited[vertex] = True
            if pred_arr[vertex] != vertex:
                temp.append(vertex)
                vertex = pred_arr[vertex]
        temp.extend(sorted_vertices)
        sorted_vertices = temp
        
    return sorted_vertices

"""
@njit
def get_tree_order(nodes_number, targets, pred_arr):
    #get nodes visiting order for flow calculation
    visited = np.zeros(nodes_number, dtype = np.bool_)
    sorted_vertices = [0] * 0
    for vertex in targets:
        temp = []
        while (not visited[vertex]):
            visited[vertex] = True
            if pred_arr[vertex] != vertex:
                temp.append(vertex)
                vertex = pred_arr[vertex]
        sorted_vertices[0:0] = temp
        
    return sorted_vertices
"""

@njit
def get_flows(nodes_number, edges_number, targets, target_flows, 
              pred_arr, sorted_vertices, pred_to_edges, t_parameter):
    flows = np.zeros(edges_number, dtype = np.float_)
    vertex_flows = np.zeros(nodes_number, dtype = np.float_)
    vertex_flows[targets] = target_flows
    
    for vertex in sorted_vertices:
        pred = pred_arr[vertex]
        edges = pred_to_edges[vertex][pred]
        edge = edges[np.argmin(t_parameter[edges])]
        flows[edge] = vertex_flows[vertex]
        vertex_flows[pred] += vertex_flows[vertex]
    return flows


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
    """
    Oracle for automatic calculations of function kGamma * \Psi (t)
    kGamma -> +0
    """

    def __init__(self, source, graph, source_correspondences, pred_to_edges):
        self.graph = graph
        self.source = source

        self.corr_targets = np.array(source_correspondences['targets'])
        self.corr_values = np.array(source_correspondences['corrs'])
        
        self.pred_to_edges = pred_to_edges
        
        self.flows = None
        self.distances = None
        
        self.time = 0
        
    def func(self, t_parameter):
        self.update_shortest_paths(t_parameter)
        return - np.dot(self.distances, self.corr_values) 

    #correct answer if func calculated flows!
    def grad(self, t_parameter):
        sorted_vertices = get_tree_order(self.graph.nodes_number, self.corr_targets, self.pred_map)
        flows = get_flows(self.graph.nodes_number, self.graph.links_number, 
                          self.corr_targets, self.corr_values, 
                          self.pred_map, sorted_vertices, self.pred_to_edges, t_parameter)
        return - flows
        
    def update_shortest_paths(self, t_parameter):
        tic = time.time()
        self.distances, self.pred_map = self.graph.shortest_distances(self.source, self.corr_targets, t_parameter)
        self.time += time.time() - tic
        
    def update_potentials(self, t_parameter):
        distances, _ = self.graph.shortest_distances(self.source, None, t_parameter)
        max_dist = np.max(distances[np.nonzero(np.isfinite(distances))])
        self.potentials = np.where(np.isinf(distances), max_dist, distances)
        self.diff_potentials = self.potentials[self.graph.terms] - self.potentials[self.graph.inits]

        
def get_pred_to_edges(graph):
    pred_to_edges = List()
    dtype = graph.edges.dtype
    numba_type = numba.from_dtype(dtype)
    for node in range(graph.nodes_number):
        temp_dict = {}
        for source, _, edge in graph.in_edges(node):
            if source in temp_dict:
                temp_dict[source].append(edge)
            else:
                temp_dict[source] = [edge]
        pred_to_edges_cur = Dict.empty(key_type = numba_type, value_type = numba_type[:])
        for source in temp_dict:
            pred_to_edges_cur[source] = np.array(temp_dict[source], dtype = dtype)
        pred_to_edges.append(pred_to_edges_cur)
        
    return pred_to_edges


class PhiBigOracle(BaseOracle):
    def __init__(self, graph, correspondences, processes_number = None):
        self.graph = graph
        self.correspondences = correspondences
        if processes_number:
            self.processes_number = processes_number
        else:
            self.processes_number = len(correspondences)
        self.t_current = self.func_current = self.grad_current = None
        
        pred_to_edges = get_pred_to_edges(graph)
        
        self.auto_oracles = []
        for source, source_correspondences in self.correspondences.items():
            self.auto_oracles.append(AutomaticOracle(source, self.graph, source_correspondences, pred_to_edges))
        self.time = 0.0
    
    def _reset(self, t_parameter):
        tic = time.time()
        self.t_current = t_parameter
        self.func_current = 0.0
        self.auto_oracles_time = 0
        for auto_oracle in self.auto_oracles:
            self.func_current += auto_oracle.func(self.t_current)
            self.auto_oracles_time += auto_oracle.time
        self.time += time.time() - tic
    
    def func(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            self._reset(t_parameter)
        return self.func_current
            
    def grad(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            self._reset(t_parameter)
        tic = time.time()
        self.t_current = t_parameter
        self.grad_current = np.zeros(self.graph.links_number)
        self.auto_oracles_time = 0
        for auto_oracle in self.auto_oracles:
            self.grad_current += auto_oracle.grad(self.t_current)
            self.auto_oracles_time += auto_oracle.time
        self.time += time.time() - tic
        return self.grad_current

    
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
