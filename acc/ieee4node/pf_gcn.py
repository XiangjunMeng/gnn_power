import torch
import numpy as np
from torch_geometric.data import Data
import csv
import cmath

DEBUG = True
#DEBUG = False

def load_data():
    raw_x = np.array([[0,0], [0,0], [0,0], [0,0]], dtype='f')
    raw_y = np.array([[0,0], [0,0], [0,0], [0,0]], dtype='f')
    raw_edge = np.array([[0,1,2], [1,2,3]], dtype=np.int_)
    #raw_edge_attr = np.array([[2000, 0.0244,0.306,0.00814,0.592,2.5,4.5,7.0,5.656854,4.272002,5.0],\
    #                          [],[]], dtype=np.int_)
    f1 = open('/home/hugo/experiment/gnn_power/acc/ieee4node/sample.player')
    f2 = open('/home/hugo/experiment/gnn_power/acc/ieee4node/node_voltages.csv')
    # load_changes = csv.reader(f1, delimiter=',')
    state_output = csv.reader(f2, delimiter=',')

    all_data = []
   
    # filling x and y using the csv files 
    for f2_row_list in state_output:
        # if DEBUG: print(f2_row_list[0][0])
        if f2_row_list[0][0] == '#':
            continue
        f1_row = f1.readline()
        if f1_row:
            f1_row_list = f1_row.split(',') 
            f1_num = complex(f1_row_list[1])
            #if DEBUG: print(f1_num)
            raw_x[3][0] = f1_num.real
            raw_x[3][1] = f1_num.imag
            f2_num = complex(f2_row_list[1])
            #if DEBUG: print(f2_num)
            raw_y[0][0] = abs(f2_num)
            raw_y[0][1] = cmath.phase(f2_num)
            f2_num = complex(f2_row_list[4])
            raw_y[1][0] = abs(f2_num)
            raw_y[1][1] = cmath.phase(f2_num)
            f2_num = complex(f2_row_list[7])
            raw_y[2][0] = abs(f2_num)
            raw_y[2][1] = cmath.phase(f2_num)
            f2_num = complex(f2_row_list[10])
            raw_y[3][0] = abs(f2_num)
            raw_y[3][1] = cmath.phase(f2_num)
            # if DEBUG: print(raw_x, raw_y)

            x = torch.from_numpy(raw_x)
            y = torch.from_numpy(raw_y)
            edge_index = torch.from_numpy(raw_edge)
            # if DEBUG: print(x, y, edge_index)
            cur_data = Data(x=x, y=y, edge_index=edge_index)
            all_data.append(cur_data)


    f1.close()
    f2.close()




    return all_data

dataset = load_data()
print(dataset[0])
