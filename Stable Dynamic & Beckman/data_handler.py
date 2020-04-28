from scanf import scanf
import re
import numpy as np
import pandas as pd

#TODO: DOCUMENTATION!!!
class DataHandler:
    def GetGraphData(self, file_name, columns_order):
        graph_data = {}
        
        with open(file_name, 'r') as myfile:
            data = myfile.read()

        graph_data['nodes number'] = scanf('<NUMBER OF NODES> %d', data)[0]
        graph_data['links number'] = scanf('<NUMBER OF LINKS> %d', data)[0]
        graph_data['zones number'] = scanf('<NUMBER OF ZONES> %d', data)[0]
        
        headlist = re.search(r'~[\s\w]+[\s]*;', data)
        print(headlist[0])
        
        my_headlist = ['Init node', 'Term node', 'Capacity', 'Free Flow Time']
        
        datalist = re.compile("[\t0-9.]+\t;").findall(data)

        datalist = [line.strip('[\t;]') for line in datalist]
        datalist = [line.split('\t') for line in datalist]

        df = pd.DataFrame(np.asarray(datalist)[:, columns_order], columns = my_headlist)
        #df = pd.DataFrame(np.asarray(datalist)[:, range(0, len(headlist))], columns = headlist)
        #df = df[list(np.array(headlist)[columns_order])]
        #print(list(np.array(headlist)[[0, 1, 2, 4]]))
        
        #init nodes
        df['Init node'] = pd.to_numeric(df['Init node'], downcast = 'integer')
        #final nodes
        df['Term node'] = pd.to_numeric(df['Term node'], downcast = 'integer')

        #capacities
        df['Capacity'] = pd.to_numeric(df['Capacity'], downcast = 'float')
        #free flow times
        df['Free Flow Time'] = pd.to_numeric(df['Free Flow Time'], downcast = 'float')

        #Table for graph ready!
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
            graph_correspondences[origin_index] = dict([scanf('%d : %f', line)
                                  for line in origin_correspondences])
        return graph_correspondences, total_od_flow

    def ReadAnswer(self, filename):
        with open(filename) as myfile:
            lines = myfile.readlines()
        lines = np.array(lines)[range(1, len(lines))]
        values_dict = {'flow': [], 'time': []}
        for line in lines:
            line = line.strip('[ \n]')
            nums = line.split(' \t')
            values_dict['flow'].append(float(nums[2]))
            values_dict['time'].append(float(nums[3]))
        return values_dict
            
