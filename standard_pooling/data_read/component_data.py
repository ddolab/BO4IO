import pandas

component_data = pandas.read_csv('data_read/csv_data/Component_Data.csv')

I = {} # input streams
k = 0
while k < len(component_data):
    current = component_data.loc[k,'name']
    index = 0
    while current[index] != ".": # find index where period is
        index += 1

    if k == len(component_data) - 1: # checks if at end of data list
        I[current[0:index]] = int(current[index + 2:len(current)])
    else:
        next = component_data.loc[k+1,'name'] # sets next component
    
    next_index = 0
    while next[next_index] != ".": # find index where period is in next component
        next_index += 1

    if int(next[next_index + 2:len(next)]) != int(current[index + 2:len(current)])+1:
        I[current[0:index]] = int(current[index + 2:len(current)])
    k += 1
    
Al = {}
Au = {}
c = {}

row = 0
while row < len(component_data):
    for key in I:
        Al_temp = {}
        Au_temp = {}
        c_temp = {}
        for i in range(I[key]):
            Al_temp[i+1] = float(component_data.loc[row,'Al'])
            Au_temp[i+1] = float(component_data.loc[row,'Au'])
            c_temp[i+1] = float(component_data.loc[row,'c'])
            row += 1
        Al[key] = Al_temp
        Au[key] = Au_temp
        c[key] = c_temp
