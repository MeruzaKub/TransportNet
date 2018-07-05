# from scipy.special import expit
#import multiprocessing as mp
from collections import defaultdict
#from scipy.misc import logsumexp
from scipy.special import expit
import numpy as np
import time
from numba import jit
import graph_tool.topology as gtt

@jit
def get_nodes_rev_tree_order(nodes_number, pred_arr):
    #get nodes visiting order for flow calculation
    visited = np.zeros(nodes_number, dtype = bool)
    sorted_vertices = []
    for vertex_index in range(nodes_number):
        temp = []
        while (not visited[vertex_index]):
            temp.append(vertex_index)
            visited[vertex_index] = True
            vertex_index = pred_arr[vertex_index]
        sorted_vertices[0:0] = temp
    return np.array(sorted_vertices)

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

        self.corr_targets = np.arange(self.graph.nodes_number)
        
        self.corr_values = np.zeros(self.graph.nodes_number)
        nonzero_corr_inidices = np.array(list(source_correspondences.keys()), dtype = 'int64') - 1
        self.corr_values[nonzero_corr_inidices] = np.array(list(source_correspondences.values()))
        assert(len(self.corr_targets) == len(self.corr_values))
        
        self.flows = None
        self.distances = None
        
        
    def func(self, t_parameter):
        self.flows, self.distances = self.shortest_paths(self.source_index,
                                     self.corr_targets, self.corr_values,
                                     t_parameter)
        return - np.dot(self.distances, self.corr_values) 

    #correct answer if func calculated flows!
    def grad(self, t_parameter):
        return - self.flows
        
    
    def shortest_paths(self, source_index, corr_targets, corr_values, t_parameter):
        #define order of vertices
        graph_tool_obj = self.graph.get_graphtool()
        ep_time_map = graph_tool_obj.new_edge_property("double",
                                                       vals = t_parameter)
        dist_map, pred_map = gtt.shortest_distance(g = graph_tool_obj,
                                                   source=graph_tool_obj.vertex(source_index),
                                                   weights = ep_time_map,
                                                   pred_map = True)

        sorted_vertices = get_nodes_rev_tree_order(self.graph.nodes_number, np.array(pred_map.a))
        #print("vertices order:")
        #print(sorted_vertices)
        #find flow values on edges
        vp_corr_map = graph_tool_obj.new_vertex_property("double",
                                                         vals = corr_values)
        ep_flows_map = graph_tool_obj.new_edge_property("double",
                                                         vals = np.zeros(self.graph.links_number))


        for vertex in sorted_vertices:
            if (graph_tool_obj.edge(pred_map[vertex], vertex)):
                ep_flows_map[graph_tool_obj.edge(pred_map[vertex], vertex)] = vp_corr_map[vertex]
                vp_corr_map[pred_map[vertex]] += vp_corr_map[vertex]

        return np.array(ep_flows_map.a), np.array(dist_map.a)
    


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
        self.grad_current = np.zeros(self.graph.links_number)
        for auto_oracle in self.auto_oracles:
            self.func_current += auto_oracle.func(self.t_current)
            self.grad_current += auto_oracle.grad(self.t_current)
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
    
         