# Attention: as shown on the table above
# nodes indexed from 1 to ...
# edges indexed from 0 to ...
#import networkx as nx
import numpy as np
import scipy.sparse as sp
import math
from numba import jitclass, int64, float64

spec = [
    ('_nodes_number', int64),
    ('_links_number', int64),
    ('_max_path_length', int64),
    ('_capacities', float64[:]),
    ('_free_flow_times', float64[:]),
    ('_sources', int64[:]),
    ('_targets', int64[:]),
    ('_in_pointers', int64[:]),
    ('_in_edges_array', int64[:]),
    ('_pred', int64[:]),
    ('_out_pointers', int64[:]),
    ('_out_edges_array', int64[:]),
    ('_succ', int64[:]),
    
]

@jitclass(spec)
class JitTransportGraph:
    
    def __init__(self, nodes_number, links_number, max_path_length, 
                 sources, targets, capacities, free_flow_times,
                 in_pointers, in_edges_array, pred,
                 out_pointers, out_edges_array, succ):
        
        self._nodes_number = nodes_number
        self._links_number = links_number
        self._max_path_length = max_path_length
        
        self._capacities = capacities
        self._free_flow_times = free_flow_times
        self._sources = sources
        self._targets = targets
        
        self._in_pointers = in_pointers
        self._in_edges_array = in_edges_array
        self._pred = pred
        self._out_pointers = out_pointers
        self._out_edges_array = out_edges_array
        self._succ = succ

    @property
    def nodes_number(self):
        return self._nodes_number
    
    @property
    def links_number(self):
        return self._links_number
      
    @property
    def max_path_length(self):
        return self._max_path_length
    
    @property
    def capacities(self):
        #return np.array(self.graph_table[['Capacity']]).flatten()
        return self._capacities
    
    @property
    def free_flow_times(self):
        #return np.array(self.graph_table[['Free Flow Time']]).flatten()
        return self._free_flow_times

    def successors(self, node_index):
        #return list(self.transport_graph.successors(vertex))
        return self._succ[self._out_pointers[node_index] : self._out_pointers[node_index + 1]]

    def predecessors(self, node_index):
        #return list(self.transport_graph.predecessors(vertex))
        return self._pred[self._in_pointers[node_index] : self._in_pointers[node_index + 1]]
        
    def in_edges(self, node_index):
        #return self._edges_indices(self.transport_graph.in_edges(vertex, data = True))
        return self._in_edges_array[self._in_pointers[node_index] : self._in_pointers[node_index + 1]]

    def out_edges(self, node_index):
        #return self._edges_indices(self.transport_graph.out_edges(vertex, data = True))
        return self._out_edges_array[self._out_pointers[node_index] : self._out_pointers[node_index + 1]]
    
    def source_of_edge(self, edge_index):
        #return self.graph_table.get_value(edge_index, 0, takeable=True)
        return self._sources[edge_index]
    
    def target_of_edge(self, edge_index):
        #return self.graph_table.get_value(edge_index, 1, takeable=True)
        return self._targets[edge_index]
    
    
class TransportGraph:
    def __init__(self, graph_data, maxpath_const = 3):
        graph_table = graph_data['graph_table']

        nodes_number = graph_data['kNodesNumber']
        links_number = graph_data['kLinksNumber']
        max_path_length = maxpath_const * int(math.sqrt(nodes_number))
        
        capacities = np.array(graph_table[['Capacity']], dtype = 'float64').flatten()
        free_flow_times = np.array(graph_table[['Free Flow Time']], dtype = 'float64').flatten()  
        sources = np.zeros(links_number, dtype = 'int64')
        targets = np.zeros(links_number, dtype = 'int64')
        
        in_incident_matrix = sp.lil_matrix((nodes_number, links_number), dtype = 'int64')
        out_incident_matrix = sp.lil_matrix((nodes_number, links_number), dtype = 'int64')
        self._nodes_indices = {}
        index = 0
        for edge, row in enumerate(graph_table[['Init node', 'Term node']].itertuples()):
            if row[1] not in self._nodes_indices:
                self._nodes_indices[row[1]] = index
                index += 1
            source = self._nodes_indices[row[1]]
            sources[edge] = source
            out_incident_matrix[source, edge] = 1
            
            if row[2] not in self._nodes_indices:
                self._nodes_indices[row[2]] = index
                index += 1
            target = self._nodes_indices[row[2]]
            targets[edge] = target
            in_incident_matrix[target, edge] = 1
        
        in_incident_matrix = in_incident_matrix.tocsr()
        in_pointers = np.array(in_incident_matrix.indptr, dtype= 'int64')
        in_edges_array = np.array(in_incident_matrix.indices, dtype= 'int64')
        pred = sources[in_edges_array]
        
        out_incident_matrix = out_incident_matrix.tocsr()
        out_pointers = np.array(out_incident_matrix.indptr, dtype= 'int64')
        out_edges_array = np.array(out_incident_matrix.indices, dtype= 'int64')
        succ = targets[out_edges_array]
            
        self._jit_graph = JitTransportGraph(nodes_number, links_number, max_path_length, 
                                            sources, targets, capacities, free_flow_times,
                                            in_pointers, in_edges_array, pred,
                                            out_pointers, out_edges_array, succ)

    @property
    def jit_graph(self):
        return self._jit_graph
        
    @property
    def nodes_number(self):
        return self._jit_graph.nodes_number
    
    @property
    def links_number(self):
        return self._jit_graph.links_number
        
    @property
    def max_path_length(self):
        return self._jit_graph.max_path_length
    
    @property
    def capacities(self):
        #return np.array(self.graph_table[['Capacity']]).flatten()
        return self._jit_graph.capacities
    
    @property
    def free_flow_times(self):
        #return np.array(self.graph_table[['Free Flow Time']]).flatten()
        return self._jit_graph.free_flow_times

    def successors(self, node_index):
        #return list(self.transport_graph.successors(vertex))
        return self._jit_graph.successors(node_index)

    def predecessors(self, node_index):
        #return list(self.transport_graph.predecessors(vertex))
        return self._jit_graph.predecessors(node_index)
    
    def get_nodes_indices(self, nodes):
        return [self._nodes_indices[node] for node in nodes]
    
    def get_node_index(self, node):
        return self._nodes_indices[node]
        
    def in_edges(self, node_index):
        #return self._edges_indices(self.transport_graph.in_edges(vertex, data = True))
        return self._jit_graph.in_edges(node_index)

    def out_edges(self, node_index):
        #return self._edges_indices(self.transport_graph.out_edges(vertex, data = True))
        return self._jit_graph.out_edges(node_index)
    
    def source_of_edge(self, edge_index):
        #return self.graph_table.get_value(edge_index, 0, takeable=True)
        return self._jit_graph.source_of_edge(edge_index)
    
    def target_of_edge(self, edge_index):
        #return self.graph_table.get_value(edge_index, 1, takeable=True)
        return self._jit_graph.target_of_edge(edge_index)
