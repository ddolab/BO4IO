import pandas
import random
import numpy as np

product_data_random = pandas.read_csv('data_read/csv_data/Product_Data.csv')

rand_seed = [2,4,6,8] # D, theta_upper, theta_lower, d

# #generate random demand values
# demand_df = product_data_random.copy()
# del demand_df['Dl']
# del demand_df['Du']
# del demand_df['d']
# #demand_df.rename(columns={'d' : 'D'})
# print(demand_df)
# random.seed(rand_seed[0])
# for i in range(demand_df.shape[0]):
#     demand_df.loc[i,'D'] = random.uniform(50,200)

#print(demand_df)
#print(product_data_random)

# randomize upper demand with theta
random.seed(rand_seed[1])
theta_upper = np.empty(product_data_random.shape[0])
for i in range(product_data_random.shape[0]):
    product_data_random.loc[i,'D'] = random.uniform(50,200)
    theta_upper[i] = random.uniform(1,1.5)
    # randomize "original" demand value for different experiments, these share same theta
    # randomize deamnd from 50-200 for all cases
    product_data_random.loc[i,'Du'] = theta_upper[i]*float(product_data_random.loc[i,'D'])
    

# randomize lower demand with theta
random.seed(rand_seed[2])
theta_lower = np.empty(product_data_random.shape[0])
for i in range(product_data_random.shape[0]):
    theta_lower[i] = random.uniform(0,0)
    product_data_random.loc[i,'Dl'] = theta_lower[i]*float(product_data_random.loc[i,'D'])

# randomize unit revenue
random.seed(rand_seed[3])
for i in range(product_data_random.shape[0]):
    product_data_random.loc[i,'d'] = float(product_data_random.loc[i,'d']) * random.uniform(0.5,1.5)


J = {} # output streams
k = 0
while k < len(product_data_random):
    current = product_data_random.loc[k,'name']
    index = 0
    while current[index] != ".": # find index where period is
        index += 1

    if k == len(product_data_random) - 1: # checks if at end of data list
        J[current[0:index]] = int(current[index + 2:len(current)])
    else:
        next = product_data_random.loc[k+1,'name'] # sets next component
    
    next_index = 0
    while next[next_index] != ".": # find index where period is in next component
        next_index += 1

    if int(next[next_index + 2:len(next)]) != int(current[index + 2:len(current)])+1:
        J[current[0:index]] = int(current[index + 2:len(current)])
    k += 1


Dl_random = {}
Du_random = {}
d_random = {}
theta_res ={}
theta_res_lower = {}
theta_res_upper = {}
D_random = {}

row = 0
while row < len(product_data_random):
    for key in J:
        Dl_temp = {}
        Du_temp = {}
        d_temp = {}
        theta_lower_temp = {}
        theta_upper_temp = {}
        D_random_temp = {}
        for i in range(J[key]):
            Dl_temp[i+1] = float(product_data_random.loc[row,'Dl'])
            Du_temp[i+1] = float(product_data_random.loc[row,'Du'])
            d_temp[i+1] = float(product_data_random.loc[row,'d'])
            theta_lower_temp[i+1] = theta_lower[row]
            theta_upper_temp[i+1] = theta_upper[row]
            D_random_temp[i+1] = float(product_data_random.loc[row,'D'])
            row += 1
        Dl_random[key] = Dl_temp
        Du_random[key] = Du_temp
        d_random[key] = d_temp
        theta_res_lower[key] = theta_lower_temp
        theta_res_upper[key] = theta_upper_temp
        theta_res['Lower Demand Theta'] = theta_res_lower
        theta_res['Upper Demand Theta'] = theta_res_upper
        D_random[key] = D_random_temp

# print('theta  res upper = ', theta_res_lower)
# print('theta  res lower = ', theta_res_upper)
#print('theta dict', theta_res)
