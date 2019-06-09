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

        graph_data['kNodesNumber'] = scanf('<NUMBER OF NODES> %d', data)[0]
        graph_data['kLinksNumber'] = scanf('<NUMBER OF LINKS> %d', data)[0]
        

        headlist = re.compile("\t"
                             "[a-zA-Z ]+"
                             "[\(\)\/\w]*").findall(data)

        
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

        #kZonesNumber = scanf('<NUMBER OF ZONES> %d', trips_data)[0]
        p = re.compile("Origin[ \t]+[\d]+")
        origins_list = p.findall(trips_data)
        origins = np.array([int(re.sub('[a-zA-Z ]', '', line)) for line in origins_list])

        p = re.compile("\n"
                       "[0-9.:; \n]+"
                       "\n\n")
        res_list = p.findall(trips_data)
        res_list = [re.sub('[\n \t]', '', line) for line in res_list]

        graph_correspondences = {}
        for origin_index in range(0, len(origins)):
            origin_correspondences = res_list[origin_index].strip('[\n;]').split(';')
            graph_correspondences[origins[origin_index]] = dict([scanf("%d:%f", line)
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
            
