import numpy as np
import pandas as pd
from pyomo.environ import (ConcreteModel, Var, NonNegativeReals, Constraint, 
                           Objective, minimize, SolverFactory, sin)
import time
import matplotlib.pyplot as plt

TOL = 1e-3

class DispatchValvePoint:
    def __init__(self, parameters: np.array, demand_profile: float, solver_framework: str, max_time: int):
        self.parameters = parameters
        self.demand_profile = demand_profile 
        self.solver_framework = solver_framework
        self.max_time = max_time
        self.n_ug = len(parameters)
        
        # x0 "padrão"
        self.initial_guess = np.array([self.parameters[i, 5] for i in range(self.n_ug)])
        
    def objective_rule(self, m):
        return sum(
            self.parameters[i, 0] +
            self.parameters[i, 1] * m.pg[i] +
            self.parameters[i, 2] * m.pg[i] ** 2 +
            abs(self.parameters[i, 3] * sin(self.parameters[i, 4] * (self.parameters[i, 5] - m.pg[i])))
            for i in range(self.n_ug)
        )
    
    def power_balance_rule(self, m):
        return sum(m.pg[i] for i in range(self.n_ug)) - self.demand_profile == TOL
    
    def optimize(self, x0):
        model = ConcreteModel()

        model.pg = Var(range(self.n_ug), domain=NonNegativeReals,
                       initialize={i: x0[i] for i in range(self.n_ug)})

        model.objective = Objective(rule=self.objective_rule, sense=minimize)

        model.power_balance = Constraint(rule=self.power_balance_rule)

        # Limites Pmin / Pmax
        for i in range(self.n_ug):
            model.pg[i].setlb(self.parameters[i, 5])
            model.pg[i].setub(self.parameters[i, 6])

        solver = SolverFactory("ipopt", executable="C:/Users/joao/anaconda3/Library/bin/ipopt.exe")
        solver.options["max_iter"] = 50000000
        solver.options["max_cpu_time"] = 360
        solver.options["tol"] = 1e-3

        result = solver.solve(model, tee=True)

        return {
            'solution': [model.pg[i].value for i in range(self.n_ug)],
            'cost': model.objective(),
            'success': result.solver.status,
            'termination_condition': result.solver.termination_condition,
            'iterations': result.solver.iterations if hasattr(result.solver, 'iterations') else None
        }


# -----------------------------
# Parâmetros e demanda
# -----------------------------
parameters = np.array([
    [21.3,   -0.3059,  0.001861,  0.0211, -3.1,  196.0, 250.0],
    [118.4,  -1.269,   0.004194,  0.1184, -13,   157.0, 230.0],
    [-2.875, -0.03389, 0.0008035, -0.003, 0.3,   332.0, 388.0],
    [266.8,  -2.338,   0.005935,  0.2668, -23,   200.0, 265.0],
    [13.92,  -0.08733, 0.001066,  0.0139, -0.9,  190.0, 338.0],
    [266.8,  -2.338,   0.005935,  0.2668, -23,   200.0, 265.0],
    [18.93,  -0.1325,  0.001107,  0.0189, -1.3,  200.0, 331.0],
    [266.8,  -2.338,   0.005935,  0.2668, -23,   200.0, 265.0],
    [14.23,  -0.01817, 0.0006121, 0.0142, -0.2,  370.0, 440.0],
    [13.97,  -0.09938, 0.001102,  0.014,  -1,    200.0, 362.0]
])
demand_profile = 2700

# Criação do modelo
dispatch_SCIP = DispatchValvePoint(parameters, demand_profile, "ipopt", 60)

# Vamos definir o x0 como Pmin para cada usina
x0 = [parameters[j, 5] for j in range(10)]

# Otimizar uma única vez
print(f"Iniciando resolução com x0 = {x0}")
start_time = time.time()
result = dispatch_SCIP.optimize(x0)
end_time = time.time()
elapsed_time = end_time - start_time

print("Resultado da otimização:")
print("Status:", result['success'])
print("Condição de Término:", result['termination_condition'])
print("Custo total:", result['cost'])
print("Solução obtida:", result['solution'])
print("Número de Iterações:", result['iterations'])
print("Tempo total de resolução: {:.2f} segundos".format(elapsed_time))
