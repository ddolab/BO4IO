import pandas
from data_read.component_data import *
from data_read.product_data import *
import numpy as np

pool_connections = pandas.read_csv('data_read/csv_data/Pool_Connections.csv')


case_studies = {}

Tx = {}
Tz = {}

row = 0
while row < len(pool_connections):
    for key in I:
        Tx_temp = []
        Tz_temp = []
        for i in range(I[key]):
            for j in range(1,pool_connections.shape[1]):
                if pool_connections.loc[row,str(j)] == 1:
                    Tx_temp.append((i+1,j))
            if not np.any(pool_connections.loc[row,'1':'8'].to_numpy()):
                for l in range(1,J[key]+1):
                    Tz_temp.append((i+1,l))
            row += 1
        Tx[key] = Tx_temp
        Tz[key] = Tz_temp



# Tz function for ease of debugging
# row = 0
# while row < len(pool_connections):
#     for key in I:
#         Tz_temp = []
#         for i in range(I[key]):
#             if not np.any(pool_connections.loc[row,'1':'8'].to_numpy()):
#                 for l in range(1,J[key]+1):
#                     Tz_temp.append((i+1,l))
#             row += 1
#         Tz[key] = Tz_temp

#print(Tz)