from scanf import scanf
import re
import numpy as np
import pandas as pd

#TODO: DOCUMENTATION!!!
class DataHandler:
    def GetGraphData(self, file_name, columns):
        graph_data = {}
        
        metadata = ''
        with open(file_name, 'r') as myfile:
            for index, line in enumerate(myfile):
                if re.search(r'^~', line) is not None:
                    skip_lines = index + 1
                    headlist = re.findall(r'[\w]+', line)
                    break
                else:
                    metadata += line
        graph_data['nodes number'] = scanf('<NUMBER OF NODES> %d', metadata)[0]
        graph_data['links number'] = scanf('<NUMBER OF LINKS> %d', metadata)[0]
        graph_data['zones number'] = scanf('<NUMBER OF ZONES> %d', metadata)[0]
        first_thru_node = scanf('<FIRST THRU NODE> %d', metadata)[0]
        
        dtypes = {'init_node' : np.int32, 'term_node' : np.int32, 'capacity' : np.float64, 'length': np.float64,
                  'free_flow_time': np.float64, 'b': np.float64, 'power': np.float64, 'speed': np.float64,'toll': np.float64,
                  'link_type' : np.int32}
        df = pd.read_csv(file_name, names = headlist, dtype = dtypes, skiprows = skip_lines, sep = r'[\s;]+', engine='python',
                         index_col = False)
        df = df[columns]
        
        df.insert(loc = list(df).index('init_node') + 1, column = 'init_node_thru', value = (df['init_node'] >= first_thru_node))
        df.insert(loc = list(df).index('term_node') + 1, column = 'term_node_thru', value = (df['term_node'] >= first_thru_node))
        graph_data['graph_table'] = df
        return graph_data
    
    
    def GetGraphCorrespondences(self, file_name):
        with open(file_name, 'r') as myfile:
            trips_data = myfile.read()
        
        total_od_flow = scanf('<TOTAL OD FLOW> %f', trips_data)[0]
        #zones_number = scanf('<NUMBER OF ZONES> %d', trips_data)[0]
        
        origins_data = re.findall(r'Origin[\s\d.:;]+', trips_data)

        graph_correspondences = {}
        for data in origins_data:
            origin_index = scanf('Origin %d', data)[0]
            origin_correspondences = re.findall(r'[\d]+\s+:[\d.\s]+;', data)
            targets = []
            corrs_vals = []
            for line in origin_correspondences:
                target, corrs = scanf('%d : %f', line)
                targets.append(target)
                corrs_vals.append(corrs)
            graph_correspondences[origin_index] = {'targets' : targets, 'corrs' : corrs_vals}
        return graph_correspondences, total_od_flow
    

    def ReadAnswer(self, filename):
        with open(filename) as myfile:
            lines = myfile.readlines()
        lines = lines[1 :]
        flows = []
        times = []
        for line in lines:
            _, _, flow, time = scanf('%d %d %f %f', line)
            flows.append(flow)
            times.append(time)
        return {'flows' : flows, 'times' : times}
            
