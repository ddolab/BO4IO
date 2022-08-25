from pyomo.environ import *
from synthetic_data import read_model, get_redox_rxn_tuple
RSet, MetSet, UB_dict, LB_dict, S_dict = read_model()
redox_rxn_tuple = get_redox_rxn_tuple(S_dict,RSet)
"""
Create concreate Pyomo model for FBA
"""
# candidate reactions in objective functions (up to 6)
obj_rxns_tup = [[(-1,'BIOMASS_Ecoli_core_w_GAM')], [(-1,'ATPM')], [(-1,'EX_glc__D_e')], [(-1,'EX_etoh_e')], redox_rxn_tuple]
obj_rxns_lst = ['l2_norm','BIOMASS_Ecoli_core_w_GAM','ATPM','EX_glc__D_e','EX_etoh_e','REDOX POTENTIAL']
C_i = {}
for i in obj_rxns_lst:
    C_i[i] = 0

mFBA = ConcreteModel()
mFBA.r = Set(initialize=RSet)
mFBA.m = Set(initialize=MetSet)
mFBA.c = Set(initialize=obj_rxns_lst)
mFBA.R = Var(mFBA.r, domain = Reals, bounds=(-1000,1000))
mFBA.S = Param(mFBA.m, mFBA.r, initialize = S_dict) #takes time
mFBA.C = Param(mFBA.c, initialize=C_i, mutable=True)
mFBA.nrxn = Param(initialize=6, mutable=True)

def mb(model, i):
    return sum(model.S[(i,j)]*model.R[j] for j in RSet) == 0
mFBA.mbcons = Constraint(mFBA.m, rule=mb)
def objFBA(m):
    return m.C[obj_rxns_lst[0]]*sum(m.R[i]**2 for i in RSet)/len(RSet)+sum(m.C[obj_rxns_lst[i]]*obj_rxns_tup[i-1][j][0]*m.R[obj_rxns_tup[i-1][j][1]] for i in range(1,value(m.nrxn)) for j in range(len(obj_rxns_tup[i-1])) )
mFBA.obj_FBA = Objective(rule=objFBA, sense=minimize)

