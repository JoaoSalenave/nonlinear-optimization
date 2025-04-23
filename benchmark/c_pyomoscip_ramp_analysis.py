import numpy as np
from pyomo.environ import (ConcreteModel, Var, NonNegativeReals, Constraint, 
                               Objective, minimize, SolverFactory, sin)
import time

class DispatchValvePointRampLimits:
    def __init__(self, parameters: np.array, demand_profile: np.array, solver_framework: str, max_time: int):
        self.parameters = parameters
        self.demand_profile = demand_profile
        self.solver_framework = solver_framework
        self.max_time = max_time
        self.n_ug = len(parameters)
        self.n_dt = len(demand_profile)
        self.initial_guess = np.array([self.parameters[i, 5] for i in range(self.n_ug)])
        self.initial_guess_dict = {(i, t): self.initial_guess[i] for i in range(self.n_ug) for t in range(self.n_dt)}

    def objective_rule(self, m):
        return sum(
            self.parameters[i, 2] +                                                                             # Constante
            self.parameters[i, 1] * m.pg[i, t] +                                                                # Linear
            self.parameters[i, 0] * m.pg[i, t] ** 2 +                                                           # Quadrático
            abs(self.parameters[i, 3] * sin(self.parameters[i, 4] * (self.parameters[i, 5] - m.pg[i, t])))      # Não-convexo
            for i in range(self.n_ug) for t in range(self.n_dt)
        )

    def power_balance_rule(self, m, t):
        return sum(m.pg[i, t] for i in range(self.n_ug)) == self.demand_profile[t]

    def lower_ramp_limit_constraint_rule(self, m, i, t):
        if t > 0:
            return m.pg[i, t] - m.pg[i, t - 1] + self.parameters[i, 8] >= 0
        else:
            return Constraint.Skip

    def upper_ramp_limit_constraint_rule(self, m, i, t):
        if t > 0:
            return m.pg[i, t] - m.pg[i, t - 1] - self.parameters[i, 7] <= 0
        else:
            return Constraint.Skip

    def optimize(self):
        model = ConcreteModel()

        model.pg = Var(range(self.n_ug), range(self.n_dt), domain=NonNegativeReals, initialize=self.initial_guess_dict)

        model.objective = Objective(rule=self.objective_rule, sense=minimize)

        model.power_balance = Constraint(range(self.n_dt), rule=self.power_balance_rule)

        model.lower_ramp_limit_constraint = Constraint(range(self.n_ug), range(self.n_dt), rule=self.lower_ramp_limit_constraint_rule)

        model.upper_ramp_limit_constraint = Constraint(range(self.n_ug), range(self.n_dt), rule=self.upper_ramp_limit_constraint_rule)

        for i in range(self.n_ug):
            for t in range(self.n_dt):
                model.pg[i, t].setlb(self.parameters[i, 5])
                model.pg[i, t].setub(self.parameters[i, 6])

        solver = SolverFactory(self.solver_framework)
        solver.options["limits/time"] = self.max_time  
        result = solver.solve(model, tee=True)

        print("Status:", result.solver.status)
        print("Condição de Término:", result.solver.termination_condition)
        print("Custo total:", model.objective())

        return model.objective(), result


def vary_ramp_limits(base_parameters, variation_factors_upper, variation_factors_lower):
    results = []
    for factor_upper, factor_lower in zip(variation_factors_upper, variation_factors_lower):
        alternative_parameters = base_parameters.copy()
        alternative_parameters[:, 7] *= factor_upper  # Variação para limite superior (UR)
        alternative_parameters[:, 8] *= factor_lower  # Variação para limite inferior (DR)
        
    
        dispatch_model = DispatchValvePointRampLimits(alternative_parameters, demand_profile, "scip", 60)
        cost, result = dispatch_model.optimize()
        results.append((factor_upper, factor_lower, cost))
    return results

# Perfil de demanda
demand_profile = np.array([
    1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776, 1924, 2072,
    2146, 2220, 2072, 1924, 1776, 1554, 1480, 1628, 1776, 2072,
    1924, 1628, 1332, 1184
])

# Parâmetros do problema
parameters = np.array([
    # c          b          a          e      f      Pmin   Pmax   UR  DR
    [0.00043,   21.60,      958.29,   450,    0.041, 150,    470,   80, 80],   # UG1
    [0.00063,   21.05,      1313.6,   600,    0.036, 135,    460,   80, 80],   # UG2
    [0.00039,   20.81,      604.97,   320,    0.028, 73,     340,   80, 80],   
    [0.00070,   23.90,      471.60,   260,    0.052, 60,     300,   50, 50],   
    [0.00079,   21.62,      480.29,   280,    0.063, 73,     243,   50, 50],   
    [0.00056,   17.87,      601.75,   310,    0.048, 57,     160,   50, 50],
    [0.00211,   16.51,      502.70,   300,    0.086, 20,     130,   30, 30],
    [0.00480,   23.23,      639.40,   340,    0.082, 47,     120,   30, 30],
    [0.10908,   19.58,      455.60,   270,    0.098, 20,      80,   30, 30],   # UG9
    [0.00951,   22.54,      692.40,   380,    0.094, 55,      55,   30, 30],   # UG10
])

variation_factors_upper = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]  # Fatores para UR 
variation_factors_lower = [1.20, 1.15, 1.10, 1.05, 1.00, 0.95, 0.90, 0.85, 0.80]  # Fatores para DR

results = vary_ramp_limits(parameters, variation_factors_upper, variation_factors_lower)

for factor_upper, factor_lower, cost in results:
    print(f"UR escala {factor_upper * 100:.0f}%, DR escala {factor_lower * 100:.0f}%: Custo = {cost}")
