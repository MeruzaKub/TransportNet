# Attention: as shown on the table above
# nodes indexed from 0 to ...
# edges indexed from 0 to ...
from graph_tool.all import *
import numpy as np
import math

    
class TransportGraph:
    def __init__(self, graph_data, maxpath_const = 3):
        graph_table = graph_data['graph_table']

        self.nodes_number = graph_data['kNodesNumber']
        self.links_number = graph_data['kLinksNumber']
        self.max_path_length = maxpath_const * int(math.sqrt(self.links_number))
        
        self.graph = Graph(directed=True)
        #nodes indexed from 0 to V-1
        vlist = self.graph.add_vertex(self.nodes_number)
        # let's create some property maps
        ep_freeflow_time = self.graph.new_edge_property("double")
        ep_capacity = self.graph.new_edge_property("double")
        
        #define data for edge properties
        self.capacities = np.array(graph_table[['Capacity']], dtype = 'float64').flatten()
        self.freeflow_times = np.array(graph_table[['Free Flow Time']], dtype = 'float64').flatten()  

        #adding edges to the graph
        inits = np.array(graph_table[['Init node']], dtype = 'int64').flatten()
        terms = np.array(graph_table[['Term node']], dtype = 'int64').flatten()
        for index in range(self.links_number):
            init_index = graph_table['Init node'][index] - 1
            term_index = graph_table['Term node'][index] - 1
            edge = self.graph.add_edge(self.graph.vertex(init_index),
                                       self.graph.vertex(term_index))
            ep_freeflow_time[edge] = self.freeflow_times[index]
            ep_capacity[edge] = self.capacities[index]
            
        #save properties to graph
        self.graph.edge_properties["freeflow_times"] = ep_freeflow_time
        self.graph.edge_properties["capacities"] = ep_capacity
       
    def get_graphtool(self):
        return self.graph

    def successors(self, node_index):
        return self.graph.get_out_neighbors(node_index)

    def predecessors(self, node_index):
        return self.graph.get_in_neighbors(node_index)
        
    #source, target and index of an edge
    def in_edges(self, node_index):
        return self.graph.get_in_edges(node_index)
    
    #source, target and index of an edge
    def out_edges(self, node_index):
        return self.graph.get_out_edges(node_index)

#    def nodes_number(self):
#        return self.nodes_number
    
#    def links_number(self):
#        return self.links_number
    
#    def capacities(self):
#        return self.capacities
    
#    def freeflow_times(self):
#        return self.freeflow_times