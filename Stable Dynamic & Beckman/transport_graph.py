# Attention: as shown on the table above
# nodes indexed from 0 to ...
# edges indexed from 0 to ...
import graph_tool.all as gt
import graph_tool.topology as gtt
import numpy as np
import math

    
class TransportGraph:
    def __init__(self, graph_table, nodes_number, links_number, maxpath_const = 3):
        self.nodes_number = nodes_number
        self.links_number = links_number
        self.max_path_length = maxpath_const * int(math.sqrt(self.links_number))
        
        self.graph = gt.Graph(directed=True)
        #nodes indexed from 0 to V-1
        vlist = self.graph.add_vertex(self.nodes_number)
        # let's create some property maps
        ep_freeflow_time = self.graph.new_edge_property("double")
        ep_capacity = self.graph.new_edge_property("double")
        
        #define data for edge properties
        self.capacities = np.array(graph_table[['capacity']], dtype = 'float64').flatten()
        self.freeflow_times = np.array(graph_table[['free_flow_time']], dtype = 'float64').flatten()  

        #adding edges to the graph
        self.inits = np.array(graph_table[['init_node']], dtype = 'int64').flatten()
        self.terms = np.array(graph_table[['term_node']], dtype = 'int64').flatten()
        for index in range(self.links_number):
            init = self.inits[index]
            term = self.terms[index]
            edge = self.graph.add_edge(self.graph.vertex(init),
                                       self.graph.vertex(term))
            ep_freeflow_time[edge] = self.freeflow_times[index]
            ep_capacity[edge] = self.capacities[index]
            
        #save properties to graph
        self.graph.edge_properties["freeflow_times"] = ep_freeflow_time
        self.graph.edge_properties["capacities"] = ep_capacity

    
    @property
    def edges(self):
        return self.graph.get_edges([self.graph.edge_index])

    def successors(self, node):
        return self.graph.get_out_neighbors(node)

    def predecessors(self, node):
        return self.graph.get_in_neighbors(node)
        
    #source, target and index of an edge
    def in_edges(self, node):
        return self.graph.get_in_edges(node, [self.graph.edge_index])
    
    #source, target and index of an edge
    def out_edges(self, node):
        return self.graph.get_out_edges(node, [self.graph.edge_index])
    
    def shortest_distances(self, source, targets, times):
        if targets is None:
            targets = np.arange(self.nodes_number)
        ep_time_map = self.graph.new_edge_property("double", vals = times)
        distances, pred_map = gtt.shortest_distance(g = self.graph,
                                                    source = source,
                                                    target = targets,
                                                    weights = ep_time_map,
                                                    pred_map = True)
        return distances, pred_map.a

#    def nodes_number(self):
#        return self.nodes_number
    
#    def links_number(self):
#        return self.links_number
    
#    def capacities(self):
#        return self.capacities
    
#    def freeflow_times(self):
#        return self.freeflow_times