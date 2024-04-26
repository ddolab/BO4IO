import pandas
from data_read.component_data import *

poolsize = pandas.read_csv('data_read/csv_data/Poolsize.csv')
L = {} # number of pools
row = 0

while row < len(poolsize)-1:
    for key in I:
        count = 0
        for i in range(poolsize.shape[1]-1):
            if poolsize.loc[row, str(i+1)] == 0 or i == poolsize.shape[1]-1:
                count
            else:
                count += 1
        row += 1
        L[key] = count


row = 0
S = {}
while row < len(poolsize):
    for key in L:
        temp = {}
        for i in range(L[key]):
            temp[i+1] = poolsize.loc[row, str(i+1)]
        S[key] = temp
        row += 1
