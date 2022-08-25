from pyomo.environ import *
from synthetic_data import read_model, get_redox_rxn_tuple
RSet, MetSet, UB_dict, LB_dict, S_dict = read_model()
redox_rxn_tuple = get_redox_rxn_tuple(S_dict,RSet)
"""
Create concreate Pyomo model for FBA
"""
# candidate reactions in objective functions (up to 6)
obj_rxns_tup = [[(-1,'BIOMASS_Ecoli_core_w_GAM')], [(-1,'ATPM')], [(-1,'EX_glc__D_e')], [(-1,'EX_etoh_e')], redox_rxn_tuple]
obj_rxns_lst = ['abs_norm','BIOMASS_Ecoli_core_w_GAM','ATPM','EX_glc__D_e','EX_etoh_e','REDOX POTENTIAL']
C_i = {}
for i in obj_rxns_lst:
    C_i[i] = 0
"""
concreate
"""
mFBA = ConcreteModel()
mFBA.r = Set(initialize=RSet)
mFBA.m = Set(initialize=MetSet)
mFBA.c = Set(initialize=obj_rxns_lst)
mFBA.R = Var(mFBA.r, domain = Reals, bounds=(-1000,1000))
mFBA.S = Param(mFBA.m, mFBA.r, initialize = S_dict) #takes time
mFBA.C = Param(mFBA.c, initialize=C_i, mutable=True)
mFBA.nrxn = Param(initialize=6, mutable=True)
mFBA.U = Var(mFBA.r, domain = NonNegativeReals)
def abs_cons_UB_rule(model,i):
    return model.R[i] <= model.U[i]
def abs_cons_LB_rule(model,i):
    return -model.R[i] <= model.U[i]
mFBA.abscons_UB = Constraint(mFBA.r, rule=abs_cons_UB_rule)
mFBA.abscons_LB = Constraint(mFBA.r, rule=abs_cons_LB_rule)
def mb(model, i):
    return sum(model.S[(i,j)]*model.R[j] for j in RSet) == 0
mFBA.mbcons = Constraint(mFBA.m, rule=mb)
def objFBA(m):
    return m.C[obj_rxns_lst[0]]*sum(m.U[i] for i in RSet)/len(RSet)+sum(m.C[obj_rxns_lst[i]]*obj_rxns_tup[i-1][j][0]*m.R[obj_rxns_tup[i-1][j][1]] for i in range(1,value(m.nrxn)) for j in range(len(obj_rxns_tup[i-1])) )
mFBA.obj_FBA = Objective(rule=objFBA, sense=minimize)


# """
# abstract
# """
# mFBA_abs = AbstractModel()
# mFBA_abs.r = Set()
# mFBA_abs.m = Set()
# mFBA_abs.c = Set(initialize=obj_rxns_lst)
# mFBA_abs.R = Var(mFBA_abs.r, domain = Reals, bounds=(-1000,1000))
# mFBA_abs.S = Param(mFBA_abs.m, mFBA_abs.r) #takes time
# mFBA_abs.C = Param(mFBA_abs.c, initialize=C_i, mutable=True)
# mFBA_abs.nrxn = Param(initialize=6, mutable=True)
# mFBA_abs.U = Var(mFBA_abs.r, domain = NonNegativeReals)
# def abs_cons_UB_rule(model,i):
#     return model.R[i] <= model.U[i]
# def abs_cons_LB_rule(model,i):
#     return -model.R[i] <= model.U[i]
# mFBA_abs.abscons_UB = Constraint(mFBA_abs.r, rule=abs_cons_UB_rule)
# mFBA_abs.abscons_LB = Constraint(mFBA_abs.r, rule=abs_cons_LB_rule)
# def mb(model, i):
#     return sum(model.S[(i,j)]*model.R[j] for j in RSet) == 0
# mFBA_abs.mbcons = Constraint(mFBA_abs.m, rule=mb)
# def objFBA(m):
#     return m.C[obj_rxns_lst[0]]*sum(m.U[i] for i in RSet)/len(RSet)+sum(m.C[obj_rxns_lst[i]]*obj_rxns_tup[i-1][j][0]*m.R[obj_rxns_tup[i-1][j][1]] for i in range(1,value(m.nrxn)) for j in range(len(obj_rxns_tup[i-1])) )
# mFBA_abs.obj_FBA = Objective(rule=objFBA, sense=minimize)
# data = {None: {
#     'r': {None: RSet},
#     'm': {None: MetSet},
#     'S': S_dict,
# }}
# mFBA = mFBA_abs.create_instance(data)

