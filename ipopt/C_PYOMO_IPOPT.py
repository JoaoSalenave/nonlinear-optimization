import numpy as np
from pyomo.environ import (ConcreteModel, Var, NonNegativeReals, Constraint, 
                           Objective, minimize, SolverFactory, sin, exp, Param, Expression, value)
import time

TOL = 1e-4
W = 1
LOSS = 0.0001

class DispatchValvePointRampLimits:
    def __init__(self, parameters: np.array, emission_parameters: np.array, demand_profile: np.array, B_matrix: np.array, solver_framework: str, max_time: int, max_iter: int):
        self.parameters = parameters
        self.emission_params = emission_parameters
        self.demand_profile = demand_profile
        self.B_matrix = B_matrix
        self.solver_framework = solver_framework
        self.max_time = max_time
        self.max_iter = max_iter
        self.n_ug = len(parameters)
        self.n_dt = len(demand_profile)
        self.initial_guess = np.array([(self.parameters[i, 5]) for i in range(self.n_ug)])
        self.initial_guess_dict = {(i, t): self.initial_guess[i] for i in range(self.n_ug) for t in range(self.n_dt)}

    def objective_rule(self, m):
        return W * sum(  
            self.parameters[i, 2] +  
            self.parameters[i, 1] * m.pg[i, t] +  
            self.parameters[i, 0] * m.pg[i, t] ** 2 +  
            abs(self.parameters[i, 3] * sin(self.parameters[i, 4] * (self.parameters[i, 5] - m.pg[i, t])))  
            for i in range(self.n_ug) for t in range(self.n_dt)
        ) + (1 - W) * sum(  
            self.emission_params[i, 0] +
            self.emission_params[i, 1] * m.pg[i, t] +
            self.emission_params[i, 2] * m.pg[i, t] ** 2 +
            self.emission_params[i, 3] * exp(self.emission_params[i, 4] * m.pg[i, t])
            for i in range(self.n_ug) for t in range(self.n_dt)
        )

    def losses_rule(self, m, t):
        return LOSS * sum(
            m.pg[i, t] * m.B[i, j] * m.pg[j, t]
            for i in range(self.n_ug) for j in range(self.n_ug)
        )

    def power_balance_rule(self, m, t):
        return sum(m.pg[i, t] for i in range(self.n_ug)) - self.demand_profile[t] - m.losses[t] == TOL

    def lower_ramp_limit_constraint_rule(self, m, i, t):
        if t > 0:
            return m.pg[i, t] - m.pg[i, t - 1] + self.parameters[i, 8] >= TOL
        else:
            return Constraint.Skip

    def upper_ramp_limit_constraint_rule(self, m, i, t):
        if t > 0:
            return m.pg[i, t] - m.pg[i, t - 1] - self.parameters[i, 7] <= TOL
        else:
            return Constraint.Skip

    def optimize(self):
        model = ConcreteModel()

        model.n_ug = self.n_ug
        model.n_dt = self.n_dt

        model.pg = Var(range(self.n_ug), range(self.n_dt), domain=NonNegativeReals, initialize=self.initial_guess_dict)

        B_dict = {(i, j): self.B_matrix[i, j] for i in range(self.n_ug) for j in range(self.n_ug)}
        model.B = Param(range(self.n_ug), range(self.n_ug), initialize=B_dict, mutable=False)

        model.losses = Expression(range(self.n_dt), rule=self.losses_rule)

        model.objective = Objective(rule=self.objective_rule, sense=minimize)

        model.power_balance = Constraint(range(self.n_dt), rule=self.power_balance_rule)

        model.lower_ramp_limit_constraint = Constraint(range(self.n_ug), range(self.n_dt), rule=self.lower_ramp_limit_constraint_rule)

        model.upper_ramp_limit_constraint = Constraint(range(self.n_ug), range(self.n_dt), rule=self.upper_ramp_limit_constraint_rule)

        for i in range(self.n_ug):
            for t in range(self.n_dt):
                model.pg[i, t].setlb(self.parameters[i, 5])
                model.pg[i, t].setub(self.parameters[i, 6])

        solver = SolverFactory("ipopt", executable="C:/Users/joao/anaconda3/Library/bin/ipopt.exe")
        solver.options["max_cpu_time"] = self.max_time
        solver.options["max_iter"] = self.max_iter
        solver.options["tol"] = 1e-6
        result = solver.solve(model, tee=True)

        print("Status:", result.solver.status)
        print("Condição de Término:", result.solver.termination_condition)
        print("Custo total:", value(model.objective))

        pg_values = np.array([[value(model.pg[i, t]) for t in range(self.n_dt)] for i in range(self.n_ug)])
        losses_values = np.array([value(model.losses[t]) for t in range(self.n_dt)])
        total_generation = pg_values.sum(axis=0)
        for t in range(self.n_dt):
            print(f"Tempo {t+1}: Geração Total = {total_generation[t]:.2f}, Demanda = {self.demand_profile[t]}, Perdas = {losses_values[t]:.2f}")

        return result


demand_profile = np.array([
    1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776, 1924, 2072,
    2146, 2220, 2072, 1924, 1776, 1554, 1480, 1628, 1776, 2072,
    1924, 1628, 1332, 1184
])

parameters = np.array([
    [0.00043, 21.60, 958.29, 450, 0.041, 150, 470, 80, 80],
    [0.00063, 21.05, 1313.6, 600, 0.036, 135, 460, 80, 80],
    [0.00039, 20.81, 604.97, 320, 0.028, 73, 340, 80, 80],
    [0.00070, 23.90, 471.60, 260, 0.052, 60, 300, 50, 50],
    [0.00079, 21.62, 480.29, 280, 0.063, 73, 243, 50, 50],
    [0.00056, 17.87, 601.75, 310, 0.048, 57, 160, 50, 50],
    [0.00211, 16.51, 502.70, 300, 0.086, 20, 130, 30, 30],
    [0.00480, 23.23, 639.40, 340, 0.082, 47, 120, 30, 30],
    [0.10908, 19.58, 455.60, 270, 0.098, 20, 80, 30, 30],
    [0.00951, 22.54, 692.40, 380, 0.094, 55, 55, 30, 30],
])

emission_params = np.array([
    [0.012,  0.085, -0.001,  0.0002,  0.03],
    [0.015,  0.095, -0.002,  0.00025, 0.035],
    [0.011,  0.080, -0.0015, 0.00015, 0.025],
    [0.013,  0.090, -0.0012, 0.0002,  0.03],
    [0.014,  0.088, -0.0013, 0.00022, 0.028],
    [0.010,  0.082, -0.0011, 0.00018, 0.026],
    [0.016,  0.094, -0.0014, 0.00024, 0.032],
    [0.017,  0.096, -0.0016, 0.00026, 0.034],
    [0.018,  0.098, -0.0017, 0.00028, 0.036],
    [0.019,  0.099, -0.0018, 0.00029, 0.037],
])

B_matrix = np.array([
    [8.70, 0.43, -4.61, 0.36, 0.32, -0.66, 0.96, -1.60, 0.80, -0.10],
    [0.43, 8.30, -0.97, 0.22, 0.75, -0.28, 5.04, 1.70, 0.54, 7.20],
    [-4.61, -0.97, 9.00, -2.00, 0.63, 3.00, 1.70, -4.30, 3.10, -2.00],
    [0.36, 0.22, -2.00, 5.30, 0.47, 2.62, -1.96, 2.10, 0.67, 1.80],
    [0.32, 0.75, 0.63, 0.47, 8.60, -0.80, 0.37, 0.72, -0.90, 0.69],
    [-0.66, -0.28, 3.00, 2.62, -0.80, 11.80, -4.90, 0.30, 3.00, -3.00],
    [0.96, 5.04, 1.70, -1.96, 0.37, -4.90, 8.24, -0.90, 5.90, -0.60],
    [-1.60, 1.70, -4.30, 2.10, 0.72, 0.30, -0.90, 1.20, -0.96, 0.56],
    [0.80, 0.54, 3.10, 0.67, -0.90, 3.00, 5.90, -0.96, 0.93, -0.30],
    [-0.10, 7.20, -2.00, 1.80, 0.69, -3.00, -0.60, 0.56, -0.30, 0.99],
]) * 1e-5 


start_time = time.time()
dispatch_model = DispatchValvePointRampLimits(parameters, emission_params, demand_profile, B_matrix, "ipopt", max_time=60, max_iter=500000)
results = dispatch_model.optimize()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de resolução: {elapsed_time:.2f} segundos")  
