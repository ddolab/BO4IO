# Abstract model of p-formulation from reference document
# import packages
from __future__ import division
import pyomo.environ as pyo
import numpy as np
import pandas as pd

# initialize model
model = pyo.AbstractModel()


# indice length
model.I = pyo.Param()
model.L = pyo.Param()
model.J = pyo.Param()
model.K = pyo.Param()


# index sets
model.i = pyo.RangeSet(1, model.I)
model.l = pyo.RangeSet(1, model.L)
model.j = pyo.RangeSet(1, model.J)
model.k = pyo.RangeSet(1, model.K)
model.Tx = pyo.Set(dimen = 2)
model.Tz = pyo.Set(dimen = 2)


# indexed parameters
model.c = pyo.Param(model.i)
model.dy = pyo.Param(model.l, model.j)
model.dz = pyo.Param(model.Tz)
model.Al = pyo.Param(model.i)
model.Au = pyo.Param(model.i)
model.S = pyo.Param(model.l)
model.Dl = pyo.Param(model.j,mutable=True)
model.Du = pyo.Param(model.j,mutable=True)
model.C = pyo.Param(model.i, model.k)
model.Pl = pyo.Param(model.j, model.k)
model.Pu = pyo.Param(model.j, model.k)

# variables
model.x = pyo.Var(model.Tx, domain=pyo.NonNegativeReals, initialize = 0.1)
model.y = pyo.Var(model.l, model.j, domain=pyo.NonNegativeReals, initialize = 0.1)
model.z = pyo.Var(model.Tz, domain=pyo.NonNegativeReals, initialize = 0.1)
model.p = pyo.Var(model.l, model.k, domain=pyo.NonNegativeReals, initialize = 0.1)


# New parameters to store the synthetic data
model.xRef = pyo.Param(model.Tx, initialize = 1,mutable=True)
model.yRef = pyo.Param(model.l, model.j, initialize = 1,mutable=True)
model.zRef = pyo.Param(model.Tz, initialize = 1,mutable=True)
model.pRef = pyo.Param(model.l, model.k, initialize = 1,mutable=True)

# define objective function
def obj_expression(m):
    return sum(m.c[i] * m.x[i,l] for (i,l) in m.Tx) - sum(m.dy[l,j] * m.y[l,j] for j in m.j for l in m.l) - (sum((m.dz[i,j] - m.c[i])*m.z[i,j] for (i,j) in m.Tz))

model.OBJ = pyo.Objective(rule=obj_expression, sense=pyo.minimize)

# constraints
def feed_avail_lower(m,i):
    return m.Al[i] <= sum(m.x[i,l] for l in m.l if (i,l) in m.Tx) + sum(m.z[i,j] for j in m.j if (i,j) in m.Tz)
model.feedavaillowerconstraint = pyo.Constraint(model.i, rule = feed_avail_lower)

def feed_avail_upper(m,i):
    return m.Au[i] >= sum(m.x[i,l] for l in m.l if (i,l) in m.Tx) + sum(m.z[i,j] for j in m.j if (i,j) in m.Tz)
model.feedavailupperconstraint = pyo.Constraint(model.i, rule = feed_avail_upper)

def pool_capacity(m,l):
    return sum(m.x[i,l] for i in m.i if (i,l) in m.Tx) <= m.S[l]
model.poolcapacity = pyo.Constraint(model.l, rule = pool_capacity)

def product_demand_lower(m,j):
    return m.Dl[j] <= sum(m.y[l,j] for l in m.l) + sum(m.z[i,j] for i in m.i if (i,j) in m.Tz)
model.demandlower = pyo.Constraint(model.j, rule = product_demand_lower)

def product_demand_upper(m,j):
    return m.Du[j] >= sum(m.y[l,j] for l in m.l) + sum(m.z[i,j] for i in m.i if (i,j) in m.Tz)
model.demandupper = pyo.Constraint(model.j, rule = product_demand_upper)

def material_balance(m,l):
    return sum(m.x[i,l] for i in m.i if (i,l) in m.Tx) - sum(m.y[l,j] for j in m.j) == 0
model.materialbalance = pyo.Constraint(model.l, rule = material_balance)

def product_quality_lower(m,j,k):
    return sum(m.p[l,k] * m.y[l,j] for l in m.l) + sum(m.C[i,k] * m.z[i,j] for i in m.i if (i,j) in m.Tz) >= m.Pl[j,k] * (sum(m.y[l,j] for l in m.l) + sum(m.z[i,j] for i in m.i if (i,j) in m.Tz))
model.qualitylower = pyo.Constraint(model.j, model.k, rule = product_quality_lower)

def product_quality_upper(m,j,k):
    return sum(m.p[l,k] * m.y[l,j] for l in m.l) + sum(m.C[i,k] * m.z[i,j] for i in m.i if (i,j) in m.Tz) <= m.Pu[j,k] * (sum(m.y[l,j] for l in m.l) + sum(m.z[i,j] for i in m.i if (i,j) in m.Tz))
model.qualityupper = pyo.Constraint(model.j, model.k, rule = product_quality_upper)

def quality_balance(m,l,k):
    return sum(m.C[i,k] * m.x[i,l] for i in m.i if (i,l) in m.Tx) == m.p[l,k] * sum(m.y[l,j] for j in m.j)
model.qualitybalance = pyo.Constraint(model.l, model.k, rule = quality_balance)

def hard_bound_y(m,l,j):
    return m.y[l,j] <= min(pyo.value(m.S[l]),pyo.value(m.Du[j]), sum(pyo.value(m.Au[i]) for (i,ll) in m.Tx))
model.hardbound_y = pyo.Constraint(model.l, model.j, rule = hard_bound_y)

def hard_bound_x(m,i,l):
    return m.x[i,l] <= min(pyo.value(m.Au[i]), pyo.value(m.S[l]), sum(pyo.value(m.Du[j]) for j in m.j))
model.hardbound_x = pyo.Constraint(model.Tx, rule = hard_bound_x)

def hard_bound_p(m,l,k):
    return m.p[l,k] <= max([pyo.value(m.C[i,k]) for i in m.i])
model.hardbound_p = pyo.Constraint(model.l, model.k, rule = hard_bound_p)

# calculate loss to the synthetic/ref data
def loss_rule(m):
    return sum((m.x[i,l]-m.xRef[i,l])**2 for (i,l) in m.Tx) +sum((m.y[l,j]-m.yRef[l,j])**2 for j in m.j for l in m.l)
model.loss = pyo.Expression(rule = loss_rule)