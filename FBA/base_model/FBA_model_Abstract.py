from pyomo.environ import *
import pandas as pd
"""
Help functions to generate the models
"""
def read_model():
    """
    Create index sets to build pyomo model of the ecoli metabolism
    """
    UB_df = pd.read_excel('base_model/model_info.xlsx', sheet_name='upper bound',index_col = 0)
    UB_dict = UB_df.to_dict('records')[0]
    LB_df = pd.read_excel('base_model/model_info.xlsx', sheet_name='lower bound',index_col = 0)
    LB_dict = LB_df.to_dict('records')[0]
    s_df = pd.read_excel('base_model/model_info.xlsx', sheet_name='stochiometric matrix',index_col = 0)
    s_dict_tmp = s_df.to_dict('index')
    RSet = s_df.columns.values.tolist()
    MetSet = s_df.index.values.tolist()
    S_dict = {}
    for i in MetSet:
        for j in RSet:
            S_dict[(i,j)] = s_dict_tmp[i][j]
    RSet_irrev_pos = []
    RSet_irrev_neg = []
    RSet_rev= []
    for idx, r in enumerate(RSet):
        if UB_dict[r] > 0 and LB_dict[r] < 0:
            RSet_rev.append(r)
        else:
            if UB_dict[r] > 0 and LB_dict[r] >=0:
                RSet_irrev_pos.append(r)
            elif UB_dict[r] <= 0 and LB_dict[r] <0:
                RSet_irrev_neg.append(r)
    return RSet,RSet_rev,RSet_irrev_pos,RSet_irrev_neg, MetSet, UB_dict, LB_dict, S_dict, 

def get_redox_rxn_tuple(S_dict,RSet):
    """
    find reactions that produce redox couple.
    return tuple of the rxn name and the stiohiometric coef of redox couples
    """
    redox_rxn_tuple = []
    redox_lst = ['nadh_c','nadph_c','fadh2_c']
    for i in redox_lst:
        for j in RSet:
            try:
                if S_dict[i,j] > 0:
                    redox_rxn_tuple.append((S_dict[i,j], j))
            except:
                pass
    return redox_rxn_tuple

def get_redox_rxn_dict(S_dict,RSet):
    """
    find reactions that produce redox couple.
    return dictionary of the rxn name (key) and its weight on the overall objective
    """
    redox_rxn_dict = {}
    redox_lst = ['nadh_c','nadph_c','fadh2_c']
    # initialize the dict
    for i in redox_lst:
        for j in RSet:
            try:
                if S_dict[i,j] > 0:
                    redox_rxn_dict[('Redox gen',j)] = 1
            except Exception as e:
                pass
    # normalize number of redox reaction 
    for key in redox_rxn_dict.keys():
        redox_rxn_dict[key] = redox_rxn_dict[key]/len(redox_rxn_dict)/10
    return redox_rxn_dict




"""
Create Abstract Pyomo model for FBA
"""

mFBA = AbstractModel()

# num of exp
mFBA.ni = Param(within=NonNegativeIntegers)

# sets
mFBA.i = RangeSet(1, mFBA.ni)
mFBA.r = Set()
mFBA.m = Set()
mFBA.c = Set() # objective rxn group set
mFBA.c_RSet = Set()
mFBA.R = Var(mFBA.i,mFBA.r, domain = Reals)
mFBA.S = Param(mFBA.m, mFBA.r) #takes time
mFBA.C = Param(mFBA.c, mutable=True)
mFBA.C_RSet = Param(mFBA.c_RSet, mutable =True)
mFBA.sigma = Param(initialize = 1,mutable=True)
mFBA.nobj = Param(mutable=True)
mFBA.Rref = Param(mFBA.i,mFBA.r, initialize = 1, mutable=True)
mFBA.Rvar = Param(mFBA.i,mFBA.r, initialize = 1, mutable=True)
mFBA.Rw = Param(mFBA.r, initialize = 1, mutable=True) #weight on Rref, 0 if Rvar = 0

# material balance
def mb(model, i, m):
    return sum(model.S[(m,j)]*model.R[i, j] for j in model.r) == 0
mFBA.mbcons = Constraint(mFBA.i,mFBA.m, rule=mb)

# loss
def loss_rule(model):
    # return sum(((model.R[i,r]-model.Rref[i,r])/model.Rref[i,r]) ** 2 for i in model.i for r in model.r if value(model.Rref[i,r])>= 0.001)/len(list(model.i)) + sum(((model.R[i,r]-model.Rref[i,r])/model.Rref[i,r]) ** 2 for i in model.i for r in model.r if value(model.Rref[i,r])<= -0.001)/len(list(model.i))#/len(list(model.r))
    # return sum(((model.R[i,r]-model.Rref[i,r])) ** 2 for i in model.i for r in model.r)/len(list(model.i))#/len(list(model.r))
    return sum((model.R[i,r]-model.Rref[i,r]) ** 2/model.Rvar[i,r] * model.Rw[r] for i in model.i for r in model.r)/len(list(model.i))/model.sigma**2#/len(list(model.r))
    # return sum((model.R[i,r]-model.Rref[i,r]) ** 2 for i in model.i for r in model.r)/len(list(model.i))/model.sigma**2#/len(list(model.r))

mFBA.loss = Expression(rule=loss_rule)

def l2_norm_rule(model):
    # return sum(sqrt(sum(model.R[i, r]**2 for r in model.r)) for i in model.i)/len(list(mcodel.r))
    return 0.005*sum((sum(model.R[i, r]**2 for r in model.r)) for i in model.i)/len(list(model.r))#/len(list(model.r))
    # return sum(sum(model.R[i, r]**2 for r in model.r)/len(list(model.r)) for i in model.i)
mFBA.l2_norm = Expression(rule=l2_norm_rule)

def l2_norm_rule_i(model, i):
    return sqrt(sum(model.R[i, r]**2 for r in model.r))
    # return sum(sum(model.R[i, r]**2 for r in model.r)/len(list(model.r)) for i in model.i)
mFBA.l2_norm_i = Expression(mFBA.i, rule=l2_norm_rule_i)

# # objective functions
#     def objFBA(m):
#         return sum(m.C[obj_rxns_lst[0]]*sum(m.R[ii, r]**2 for r in RSet)/len(RSet)+sum(m.C[obj_rxns_lst[i]]*obj_rxns_tup[i-1][j][0]*m.R[ii,obj_rxns_tup[i-1][j][1]] for i in range(1,value(m.nobj)) for j in range(len(obj_rxns_tup[i-1])) ) for ii in m.i)


# normalization factor
# mFBA.Cn = Param(mFBA.c, mutable=True)

# objective functions
def objFBA(m):
    # return m.C[obj_rxns_lst[0]]*m.l2_norm + sum(sum(m.C[obj_rxns_lst[i]]*mFBA.Ci*m.R[ii,obj_rxns_tup[i-1][j][1]] for i in range(1,value(m.nobj)) for j in range(len(obj_rxns_tup[i-1])) ) for ii in m.i)
    # return (1-sum(m.C[c] for c in m.c))*m.l2_norm + sum(sum(m.Cn[c]*m.C[c]*m.R[ii,c] for c in m.c) for ii in m.i)
    return m.l2_norm + sum(sum(m.C[i]*m.C_RSet[i,j]*m.R[ii,j] for i,j in m.c_RSet) for ii in m.i)
# m.C[obj_rxns_lst[0]]*m.l2_norm + sum(sum(m.C[obj_rxns_lst[i]]*mFBA.Ci*m.R[ii,obj_rxns_tup[i-1][j][1]] for i in range(1,value(m.nobj)) for j in range(len(obj_rxns_tup[i-1])) ) for ii in m.i)
    
mFBA.obj_FBA = Objective(rule=objFBA, sense=minimize)