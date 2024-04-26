import pandas
import random
import numpy as np

# product_data_random = pandas.read_csv('BO4IO_Blending_FP/data_read/csv_data/Product_Data.csv')

# random.seed(10)
# theta = np.empty(product_data_random.shape[0])
# for i in range(product_data_random.shape[0]):
#     theta[i] = random.uniform(0.5,1.5)
#     print(theta[i])

y= {(1, 1): 24.892534190382236, (1, 2): 117.18662262634645}

key_list = list(y.keys())

print(key_list[0][0])