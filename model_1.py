from __future__ import division
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


model = pyo.ConcreteModel()


model.n = pyo.Param(within=pyo.NonNegativeReals)
d = {1: 80, 2: 270, 3: 250}

model.I = [1, 2, 3]
model.J = [1, 2, 3]

model.D = pyo.Set(initialize=d.keys())

model.С = pyo.Param(initialize=5)


distances = {(1, 1): 0,    (1, 2): 6,    (1, 3): 9,
     (2, 1): 6,    (2, 2): 0,    (2, 3): 7,
     (3, 1): 6,    (3, 2): 3,    (3, 3): 0,
}

model.d = pyo.Param(model.I, model.J, initialize=distances, within=pyo.NonNegativeReals)

model.x = pyo.Var(model.I, model.J, within=pyo.Binary)
model.q = pyo.Var(model.I, model.J, domain=pyo.NonNegativeReals)


def obj_expression(m):
    return pyo.summation(m.d, m.x)


model.OBJ = pyo.Objective(expr=obj_expression, sense=pyo.minimize)


def constraint_xij(m, i):
    return sum(m.x[i, j] for j in m.J) == 1


def constraint_xji(m, j):
    return sum(m.x[j, i] for i in m.I) == 1


model.constraint_xij = pyo.Constraint(model.I, rule=constraint_xij)
model.constraint_xji = pyo.Constraint(model.I, rule=constraint_xji)


def constraint_qji_qij(m, i, j):
    qji_sum = sum(m.q[j, i] for i in m.I)
    qij_sum = sum(m.q[i, j] for j in m.J)
    return qji_sum - qij_sum == m.D[i]


model.constraint_qji_qij = pyo.Constraint(model.I, model.J, rule=constraint_qji_qij)


def constraint_qij_greater(m, i, j):
    return (m.q[i, j]) >= 0


model.constraint_qij_greater = pyo.Constraint(model.I, model.J, rule=constraint_qij_greater)


def constraint_qij_lower(m, i, j):
    return (m.q[i, j]) <= m.С*(m.x[i, j])


model.constraint_qij_lower = pyo.Constraint(model.I, model.J, rule=constraint_qij_lower)
List = list(model.x.keys())
for i in List:
    if model.x[i]() != 0:
        print(i, '---', distances.get(i))

opt = pyo.SolverFactory('cplex')
opt.solve(model, tee=False)