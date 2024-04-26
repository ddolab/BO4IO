import pandas
from data_read.component_data import *

component_quality = pandas.read_csv('data_read/csv_data/Component_Quality.csv')
K = {} 
row = -1

while row < len(component_quality)-1:
    for key in I:
        count = 0
        row += I[key]
        for i in range(component_quality.shape[1]-1):
            if component_quality.loc[row, str(i)] == 0 or i == component_quality.shape[1]-1:
                count
            else:
                count += 1
        K[key] = count


row = 0
C = {}
while row < len(component_quality):
    for key in K:
        temp = {}
        for i in range(I[key]):
            for k in range(K[key]):
                temp[(i+1,k+1)] = component_quality.loc[row,str(k)]
            row += 1
        C[key] = temp


