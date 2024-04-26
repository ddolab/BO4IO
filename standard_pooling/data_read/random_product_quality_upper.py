import pandas
import random
from data_read.product_data import *
from data_read.component_quality import *

product_quality_upper = pandas.read_csv('data_read/csv_data/Product_Quality_Upper.csv')
upper_quality = {} # 

k = 0
while k < len(product_quality_upper):
    current = product_quality_upper.loc[k,'name']
    index = 0
    while current[index] != ".": # find index where period is
        index += 1

    if k == len(product_quality_upper) - 1: # checks if at end of data list
        upper_quality[current[0:index]] = int(current[index + 2:len(current)])
    else:
        next = product_quality_upper.loc[k+1,'name'] # sets next component
    
    next_index = 0
    while next[next_index] != ".": # find index where period is in next component
        next_index += 1

    if int(next[next_index + 2:len(next)]) != int(current[index + 2:len(current)])+1:
        upper_quality[current[0:index]] = int(current[index + 2:len(current)])
    k += 1


random.seed(4)
row = 0
Pu_random = {}
while row < len(product_quality_upper):
    for key in upper_quality:
        temp = {}
        for j in range(J[key]):
            for k in range(K[key]):
                temp[(j+1,k+1)] = random.uniform(5,1) #product_quality_upper.loc[row,str(k)] * random.uniform(0.5,1.5)
            row += 1
        Pu_random[key] = temp



