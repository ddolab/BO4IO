import pandas
import random
import numpy as np

product_data = pandas.read_csv('data_read/csv_data/Product_Data.csv')

rand_seed = 8

random.seed(rand_seed)
theta = np.empty(product_data.shape[0])
for i in range(product_data.shape[0]):
    theta[i] = random.uniform(0.5,1.5)

J = {} # output streams
k = 0
while k < len(product_data):
    current = product_data.loc[k,'name']
    index = 0
    while current[index] != ".": # find index where period is
        index += 1

    if k == len(product_data) - 1: # checks if at end of data list
        J[current[0:index]] = int(current[index + 2:len(current)])
    else:
        next = product_data.loc[k+1,'name'] # sets next component
    
    next_index = 0
    while next[next_index] != ".": # find index where period is in next component
        next_index += 1

    if int(next[next_index + 2:len(next)]) != int(current[index + 2:len(current)])+1:
        J[current[0:index]] = int(current[index + 2:len(current)])
    k += 1


Dl = {}
Du = {}
d = {}

row = 0
while row < len(product_data):
    for key in J:
        Dl_temp = {}
        Du_temp = {}
        d_temp = {}
        for i in range(J[key]):
            Dl_temp[i+1] = float(product_data.loc[row,'Dl'])
            Du_temp[i+1] = float(product_data.loc[row,'Du'])
            d_temp[i+1] = float(product_data.loc[row,'d']) 
            row += 1
        Dl[key] = Dl_temp
        Du[key] = Du_temp
        d[key] = d_temp


