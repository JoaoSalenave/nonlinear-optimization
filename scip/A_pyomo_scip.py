import numpy as np
from pyomo.environ import *

# Data
n_ug = 10

abc = np.array([
    [26.97, -0.3975, 0.002176],
    [118.4, -1.269, 0.004194],
    [-2.875, -0.03389, 0.0008035],
    [266.8, -2.338, 0.005935],
    [13.92, -0.08733, 0.001066],
    [266.8, -2.338, 0.005935],
    [18.93, -0.1325, 0.001107],
    [266.8, -2.338, 0.005935],
    [88.53, -0.5675, 0.001554],
    [13.97, -0.09938, 0.001102],
])

bounds = np.array([
    (100.0, 196.0),
    (157.0, 230.0),
    (332.0, 388.0),
    (200.0, 265.0),
    (190.0, 338.0),
    (200.0, 265.0),
    (200.0, 331.0),
    (200.0, 265.0),
    (213.0, 370.0),
    (200.0, 362.0),
])

demand = 2500


model = ConcreteModel()

model.pg = Var(range(n_ug), domain=NonNegativeReals)

def objective_rule(m):
    return sum(abc[i, 0] + abc[i, 1] * m.pg[i] + abc[i, 2] * m.pg[i] ** 2 for i in range(n_ug))

model.objective = Objective(rule=objective_rule, sense=minimize)


def power_balance_rule(m):
    return sum(m.pg[i] for i in range(n_ug)) == demand

model.power_balance = Constraint(rule=power_balance_rule)


for i in range(n_ug):
    model.pg[i].setlb(bounds[i, 0])
    model.pg[i].setub(bounds[i, 1])


solver = SolverFactory('scip')
result = solver.solve(model, tee=True)

print("Status:", result.solver.status)
print("Termination condition:", result.solver.termination_condition)
print("Optimal value:", model.objective())
print("Optimal solution:")
