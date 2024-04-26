import pandas
import random
from data_read.product_data import *
from data_read.component_quality import *

product_quality_lower = pandas.read_csv('data_read/csv_data/Product_Quality_Lower.csv')
lower_quality = {}

k = 0
while k < len(product_quality_lower):
    current = product_quality_lower.loc[k,'name']
    index = 0
    while current[index] != ".": # find index where period is
        index += 1

    if k == len(product_quality_lower) - 1: # checks if at end of data list
        lower_quality[current[0:index]] = int(current[index + 2:len(current)])
    else:
        next = product_quality_lower.loc[k+1,'name'] # sets next component
    
    next_index = 0
    while next[next_index] != ".": # find index where period is in next component
        next_index += 1

    if int(next[next_index + 2:len(next)]) != int(current[index + 2:len(current)])+1:
        lower_quality[current[0:index]] = int(current[index + 2:len(current)])
    k += 1



random.seed(5)
row = 0
Pl_random = {}
while row < len(product_quality_lower):
    for key in lower_quality:
        temp = {}
        for j in range(J[key]):
            for k in range(K[key]):
                temp[(j+1,k+1)] = product_quality_lower.loc[row,str(k)] * random.uniform(0.5,1.5)
            row += 1
        Pl_random[key] = temp

