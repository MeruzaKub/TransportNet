#import multiprocessing as mp
from collections import defaultdict
#from scipy.misc import logsumexp
import numpy as np
import time
from numba import njit
import graph_tool.topology as gtt


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

@njit
def get_flows(nodes_number, edges_number, targets, target_flows, pred_arr, pred_edges, sorted_vertices):  
    flows = np.zeros(edges_number, dtype = np.float_)
    vertex_flows = np.zeros(nodes_number, dtype = np.float_)
    vertex_flows[targets] = target_flows
    for i, vertex in enumerate(sorted_vertices):
        edge = pred_edges[i]
        flows[edge] = vertex_flows[vertex]
        pred = pred_arr[vertex]
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
        

class AutomaticOracle(BaseOracle):
    """
    Oracle for automatic calculations of function kGamma * \Psi (t)
    kGamma -> +0
    """

    def __init__(self, source, graph, source_correspondences):
        self.graph = graph
        self.source_index = source - 1

        self.corr_targets = np.array(list(source_correspondences.keys()), dtype = 'int64') - 1
        self.corr_values = np.array(list(source_correspondences.values()))
        
        self.flows = None
        self.distances = None
        
    def func(self, t_parameter):
        self.update_shortest_paths(t_parameter)
        return - np.dot(self.distances, self.corr_values) 

    #correct answer if func calculated flows!
    def grad(self, t_parameter):
        sorted_vertices = get_tree_order(self.graph.nodes_number, self.corr_targets, self.pred_map)
        pred_edges = [self.graph.pred_to_edge[vertex][self.pred_map[vertex]] for vertex in sorted_vertices]
        flows = get_flows(self.graph.nodes_number, self.graph.links_number, 
                          self.corr_targets, self.corr_values, self.pred_map, pred_edges, sorted_vertices)
        return - flows
        
    
    def update_shortest_paths(self, t_parameter):
        #define order of vertices
        graph_tool_obj = self.graph.get_graphtool()
        ep_time_map = graph_tool_obj.new_edge_property("double",
                                                       vals = t_parameter)
        self.distances, pred_map = gtt.shortest_distance(g = graph_tool_obj,
                                                         source = self.source_index,
                                                         target = self.corr_targets,
                                                         weights = ep_time_map,
                                                         pred_map = True)
        self.pred_map = pred_map.a


class PhiBigOracle(BaseOracle):
    def __init__(self, graph, correspondences, processes_number = None):
        self.graph = graph
        self.correspondences = correspondences
        if processes_number:
            self.processes_number = processes_number
        else:
            self.processes_number = len(correspondences)
        self.t_current = None
        self.func_current = None
        self.grad_current = None
        
        self.auto_oracles = []
        for source, source_correspondences in self.correspondences.items():
            self.auto_oracles.append(AutomaticOracle(source, self.graph, source_correspondences))
        self.time = 0.0
    
    def _reset(self, t_parameter):
        #print('Start reset')
        tic = time.time()
        self.t_current = t_parameter
        self.func_current = 0.0
        for auto_oracle in self.auto_oracles:
            self.func_current += auto_oracle.func(self.t_current)
        self.time += time.time() - tic
        #print('Stop reset')
    
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
        for auto_oracle in self.auto_oracles:
            self.grad_current += auto_oracle.grad(self.t_current)
        self.time += time.time() - tic
        return self.grad_current
    
         