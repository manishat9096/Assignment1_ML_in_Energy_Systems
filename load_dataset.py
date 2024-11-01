import pandas as pd
import numpy as np
import gurobipy as gb
from gurobipy import GRB

#load all datasets from the excel files

production_cost = pd.read_csv('Datasets/Production cost of generating units.csv')
start_up_cost = pd.read_csv('Datasets/Start-up cost of generating units.csv')
demand = pd.read_csv('Datasets/Loads.csv')
B = pd.read_csv('Datasets/B (power transfer factor of each bus to each line).csv', sep= ';')
line_capacity = pd.read_csv('Datasets/Transmission capacity of lines.csv')
ramping_rate_gen = pd.read_csv('Datasets/Ramping rate of generating units.csv')
wind_capacity = pd.read_csv('Datasets/Capacity of wind farms.csv')
max_prod_limit = pd.read_csv('Datasets/Maximum production of generating units.csv')
min_prod_limit = pd.read_csv('Datasets/Minimum production of generating units.csv')
min_up_time = pd.read_csv('Datasets/Minimum up time of generating units.csv')
min_down_time = pd.read_csv('Datasets/Minimum down time of generating units.csv')


# define length of dataset
conventional_gen = ['G1', 'G2', 'G3']
wind_gen = ['W1', 'W2']
load_demand = ['D1', 'D2', 'D3']
lines = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']
buses = B.columns

production_cost.index = conventional_gen
start_up_cost.index = conventional_gen
max_prod_limit.index = conventional_gen
min_prod_limit.index = conventional_gen
min_up_time.index = conventional_gen
min_down_time.index = conventional_gen
ramping_rate_gen.index = conventional_gen
demand.index = load_demand
wind_capacity.index = wind_gen
line_capacity.index = lines
B.index = lines

# define the network topology - connect lines to bus
Time = list(range(0, 24))

gen_to_bus = {'G1': 'Bus1','G2': 'Bus2','G3': 'Bus6','W1': 'Bus4','W2': 'Bus5'}
bus_to_gen = {'Bus1': 'G1','Bus2': 'G2','Bus6': 'G3','Bus4': 'W1','Bus5': 'W2'}
demand_to_bus = {'D1': 'Bus4','D2': 'Bus5','D3': 'Bus6'}
bus_to_demand = {'Bus4': 'D1', 'Bus5': 'D2', 'Bus6': 'D3'}

#define the optimization problem
model = gb.Model('Unit Commitment Problem')

#define variables
power_gen = { (g, t): model.addVar(lb = 0, name = f"Power generated in Gen {g} at time {t}")
             for g in conventional_gen + wind_gen
             for t in Time  
            }

commitment = { (g, t): model.addVar(vtype= GRB.BINARY, name = f"Commitment of Gen {g} at time {t}")
              for g in conventional_gen + wind_gen
             for t in Time
             }

start_up_gen_cost = { (g, t): model.addVar(lb = 0, name = f"Start up cost of Gen {g} at time {t}")
                    for g in conventional_gen
                    for t in Time
                    }

for g in wind_gen:
    production_cost.loc[g] = 0

# define the objective function
cost1 =  gb.quicksum(production_cost['cost_op'][g] * power_gen[g, t] for g in conventional_gen + wind_gen for t in Time)
cost2 = gb.quicksum(start_up_gen_cost[g, t] for g in conventional_gen for t in Time)
model.setObjective(cost1 + cost2, GRB.MINIMIZE)

# define the constraints

# constraint 1: Conventional generation within minimun and maximum limits
model.addConstrs((power_gen[g, t] >= min_prod_limit['pgmin'][g])
                 for g in conventional_gen 
                 for t in Time)

model.addConstrs((power_gen[g, t] <= max_prod_limit['pgmax'][g])
                 for g in conventional_gen 
                 for t in Time)

# constraint 2: power balance equation
model.addConstr(gb.quicksum(power_gen[g, t] for g in conventional_gen + wind_gen for t in Time) 
                == gb.quicksum(demand['load'][l] for l in load_demand))

#constraint 3: Minimum and maximum line capacity limits are enforced
for l in lines:
    for t in Time:
        flow = gb.quicksum(
            B[n][l] * 
            (
                (power_gen[bus_to_gen.get(n), t] if gen_to_bus.get(n) in conventional_gen + wind_gen else 0)
                - (demand['load'][bus_to_demand.get(n)] if bus_to_demand.get(n) in demand['load'] else 0) 
            )
            for n in buses
        )
        model.addConstr(flow <= line_capacity['fmax'][l], name=f"Flow_upper_L{l}_T{t}")
        model.addConstr(flow >= -line_capacity['fmax'][l], name=f"Flow_lower_L{l}_T{t}")

#constraint 4: Start cost of generator is considered if started within the hour
model.update()