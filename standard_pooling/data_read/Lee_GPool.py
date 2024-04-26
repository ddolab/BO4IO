import pandas as pd
import numpy
d = pd.read_excel('data_read/csv_data/Lee_GPool.xlsx',sheet_name="dj", header=None)
d = {i+1: item for i, item in enumerate(list(d.to_dict()[0].values()))}
D = pd.read_excel('data_read/csv_data/Lee_GPool.xlsx',sheet_name="D", header=None)
D = {i+1: item for i, item in enumerate(list(D.to_dict()[0].values()))}
C = pd.read_excel('data_read/csv_data/Lee_GPool.xlsx',sheet_name="Cik", header=None).dropna(axis=1)
C = {(i+1,k+1): C[k][i] for k in C.keys() for i in C[k].keys()}
cr = pd.read_excel('data_read/csv_data/Lee_GPool.xlsx',sheet_name="cil", header=None).dropna(axis=1)
I = cr.shape[0]
L = cr.shape[1]
cr = {(i+1,l+1): cr[l][i] for l in cr.keys() for i in cr[l].keys()}
crinit = pd.read_excel('data_read/csv_data/Lee_GPool.xlsx',sheet_name="crinit", header=None).dropna(axis=1)
crinit = {i+1: item for i, item in enumerate(list(crinit.to_dict()[0].values()))}
crpool = pd.read_excel('data_read/csv_data/Lee_GPool.xlsx',sheet_name="crpool", header=None).dropna(axis=1)
crpool = {i+1: item for i, item in enumerate(list(crpool.to_dict()[0].values()))}


Pl = pd.read_excel('data_read/csv_data/Lee_GPool.xlsx',sheet_name="PL", header=None).dropna(axis=1)
J = Pl.shape[0]
K = Pl.shape[1]
Pl = {(j+1,k+1): Pl[k][j] for k in Pl.keys() for j in Pl[k].keys()}
Pu = Pl.copy()
Tx = [(i+1,l+1) for i in range(I) for l in range(L)]
Tt = [(i+1, j+1) for i in range(L) for j in range(L) if i != j]
Tz = []
c = {i+1: 0 for i in range(I)}
Au = {i+1: 1000 for i in range(I)}
Al = {i+1: 0 for i in range(I)}
S = {l+1: 1000 for l in range(L)}

max_Du = max(D.values())
min_Du = min(D.values())
# print(max_Du)
# print(min_Du)


# x_next = numpy.random.dirichlet(alpha=[1]*int(n_dim*2))*numpy.array([0.3]*int(n_dim*2)) + numpy.array([0.2]*int(n_dim)+[0.7]*int(n_dim))
