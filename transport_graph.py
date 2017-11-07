# Attention: as shown on the table above
# nodes indexed from 1 to ...
# edges indexed from 0 to ...
import networkx as nx
import numpy as np
import math

class TransportGraph:
    def __init__(self, graph_data, maxpath_const = 3):
        self.graph_table = graph_data['graph_table']
        
        self.kNodesNumber = graph_data['kNodesNumber']
        self.kLinksNumber = graph_data['kLinksNumber']
        self.kMaxPathLength = maxpath_const * int(math.sqrt(self.kNodesNumber))
      
        self.transport_graph = nx.DiGraph()
        
        self.transport_graph.add_nodes_from(np.arange(1, self.kNodesNumber + 1))

        for link_index in range(0, self.kLinksNumber):
            self.transport_graph.add_edge(self.graph_table.get_value(link_index, 0, takeable=True), 
                                          self.graph_table.get_value(link_index, 1, takeable=True),
                                          edge_index = link_index)
    

    def capacities(self):
        return np.array(self.graph_table[['Capacity']]).flatten()
        
    def freeflowtimes(self):
        return np.array(self.graph_table[['Free Flow Time']]).flatten()

    def nodes(self):
        return self.transport_graph.nodes()
      
    def edges(self):
        return range(0, self.transport_graph.number_of_edges())

    def successors(self, vertex):
        return self.transport_graph.successors(vertex)

    def predecessors(self, vertex):
        return self.transport_graph.predecessors(vertex)
      
    def edge_index(self, source, term_vertex):
        return self.transport_graph[source][term_vertex]['edge_index']
    
    def in_edges(self, vertex):
        return self._edges_indices(self.transport_graph.in_edges(vertex, data = True))

    def out_edges(self, vertex):
        return self._edges_indices(self.transport_graph.out_edges(vertex, data = True))

    def _edges_indices(self, edges):
        return [element[2]['edge_index'] for element in edges] 
      
    def source_of_edge(self, edge_index):
        return self.graph_table.get_value(edge_index, 0, takeable=True)
      
    def target_of_edge(self, edge_index):
        return self.graph_table.get_value(edge_index, 1, takeable=True)
        
