import numpy as np
from pyomo.environ import(ConcreteModel, Var, NonNegativeReals, Constraint,
                              Objective, minimize, SolverFactory, sin)
import time

start_time = time.time()

class DispatchValvePoint:
        def __init__(self, parameters: np.array, demand_profile: np.array, solver_framework: str, max_time: int):
            self.parameters = parameters
            self.demand_profile = demand_profile 
            self.solver_framework = solver_framework
            self.max_time = max_time
            self.n_ug = len(parameters)
            self.initial_guess = np.array([ self.parameters[i, 5]
                                          for i in range(self.n_ug)])
            
        def objective_rule(self, m):
            return sum(
                self.parameters[i, 0] +
                self.parameters[i, 1] * m.pg[i] +
                self.parameters[i, 2] * m.pg[i] ** 2 +
                abs(self.parameters[i, 3] * sin(self.parameters[i, 4] * (self.parameters[i, 5] - m.pg[i])))
                for i in range(self.n_ug)
            )
        
        def power_balance_rule(self, m):
            return sum(m.pg[i] for i in range(self.n_ug)) == self.demand_profile
        
        def optimize(self):
            model = ConcreteModel()

            model.pg = Var(range(self.n_ug), domain=NonNegativeReals, initialize=self.initial_guess)

            model.objective = Objective(rule=self.objective_rule, sense=minimize)

            model.power_balance = Constraint(rule=self.power_balance_rule)

            for i in range(self.n_ug):
                model.pg[i].setlb(self.parameters[i, 5])
                model.pg[i].setub(self.parameters[i, 6])

            solver = SolverFactory(self.solver_framework)
            solver.options["limits/time"] = self.max_time
            result = solver.solve(model, tee=True)

            print("Status:", result.solver.status)
            print("Condição de Término:", result.solver.termination_condition)
            print("Custo total:", model.objective())
            print("Despachos:")
            for i in range(self.n_ug):
                print(f"  pg[{i + 1}] = {model.pg[i]()} custo[{i + 1}] = {parameters[i, 0] + model.pg[i]()*parameters[i, 1] + model.pg[i]()**2*parameters[i, 2] + abs(parameters[i, 3] * sin(parameters[i, 4] * (parameters[i, 5] - model.pg[i]()))) }")

            return result

demand_profile = 2700.0            

parameters = np.array([
    #a          #b              #c              #e          #f      #Pmin   #Pmax
    [21.3,      -0.3059,        0.001861,       0.0211,     -3.1,   196.0,  250.0],   #UG1  
    [118.4,     -1.269,         0.004194,       0.1184,     -13,    157.0,  230.0],   #UG2
    [-2.875,    -0.03389,       0.0008035,      -0.003,     0.3,    332.0,  388.0], 
    [266.8,     -2.338,         0.005935,       0.2668,     -23,    200.0,  265.0], 
    [13.92,     -0.08733,       0.001066,       0.0139,     -0.9,   190.0,  338.0],
    [266.8,     -2.338,         0.005935,       0.2668,     -23,    200.0,  265.0],   # . . .
    [18.93,     -0.1325,        0.001107,       0.0189,     -1.3,   200.0,  331.0], 
    [266.8,     -2.338,         0.005935,       0.2668,     -23,    200.0,  265.0], 
    [14.23,     -0.01817,       0.0006121,      0.0142,     -0.2,   370.0,  440.0],   #UG9
    [13.97,     -0.09938,       0.001102,       0.014,      -1,     200.0,  362.0],   #UG10
    ])

    #Resultados
dispatch_SCIP = DispatchValvePoint(parameters, demand_profile, "scip", 60)
results_SCIP = dispatch_SCIP.optimize()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
