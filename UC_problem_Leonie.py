# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:33:02 2024

@author: leoni
"""

import numpy as np
import gurobipy as gb
import pandas as pd
import random

P_max = pd.read_csv('./Maximum production of generating units.csv')
P_max_wind= pd.read_csv('./Capacity of wind farms.csv')
#Combine P_max of conv and wind
#P_max_conv.rename(columns={'pgmax': 'pmax'}, inplace=True)
#P_max_wind.rename(columns={'wind': 'pmax'}, inplace=True)

#P_max = pd.concat([P_max_conv, P_max_wind], ignore_index=True)


P_min = pd.read_csv('./Minimum production.csv')
R = pd.read_csv('./Ramping rate.csv')
Up_time = pd.read_csv('./Minimum up time.csv')
Dw_time = pd.read_csv('./Minimum down time.csv')    
Demand = pd.read_csv('./Loads.csv')   
C_op = pd.read_csv('./Production cost.csv') 
C_st = pd.read_csv('./Start-up cost.csv') 
B = pd.read_csv('./B (power transfer factor of each bus to each line).csv') 
F_max = pd.read_csv('./Transmission capacity of lines.csv')
WINDFARMS =  ['WF1', 'WF2'] 
BUSES = [f'Bus{i}' for i in range(1,7)]

data_generators = [
    (1, 'G1'), (2, 'G2'), (6, 'G3'),
    (4, 'WF1'), (5, 'WF2')
]

data_loads = [
    (4, 'L1'), (5, 'L2'), (6, 'L3'),
]

data_buses = [ (i, f'Bus{i}') for i in range(1,len(BUSES)+1)]

# Create a dictionary that maps each generator/load/bus to its corresponding node
generator_nodes = {generator: node for node, generator in data_generators}
load_nodes = {load: node for node, load in data_loads}
bus_nodes = {bus: node for node, bus in data_buses}


Bus_gen = pd.DataFrame({
    'G1': [1, 0, 0, 0, 0, 0],  # Generation at Bus1
    'G2': [0, 1, 0, 0, 0, 0],  # Generation at Bus2
    'G3': [0, 0, 0, 0, 0, 1]   # Generation at Bus3
})
Busgen_matrix=Bus_gen.values


Bus_dem = pd.DataFrame({
    'D1': [0, 0, 0, 1, 0, 0],  # Demand at Bus1
    'D2': [0, 0, 0, 0, 1, 0],  # Demand at Bus2
    'D3': [0, 0, 0, 0, 0, 1]   # Demand at Bus3
})
Busdem_matrix=Bus_dem.values

B = pd.DataFrame(B)
Demand = pd.DataFrame(Demand)
Demand = Demand.values

B = np.vstack([list(map(float, row[0].split(';'))) for row in B.values])


Hg=np.dot(B,Bus_gen)
Hl=np.dot(B,Bus_dem)


n_g=len(P_max)
n_w=len(P_max_wind)
n_l=len(F_max)
n_d=len(Demand)
n_n=len(Bus_gen)
n_t=24

#Line definiton
lines_def = [
    (1, 2), 
    (2, 3), 
    (1, 4), 
    (2, 4),  
    (4, 5),
    (5, 6),
    (3, 6)
]

for line in range(len(lines_def)):
    lines_def[line] = tuple(list(lines_def[line]) + [F_max['fmax'][line]])
    
n_samples=5 
  
# Create dictionaries to store data samples
loads_samples = {f"L{i+1}": [] for i in range(len(data_loads))}
windfarm_prod = {f"WF{i+1}": [] for i in range(len(P_max_wind))}

# Populate samples for each wind farm
for wf_index, wf_name in enumerate(windfarm_prod.keys()):
    max_capacity = P_max_wind['wind'][wf_index]
    for i in range(n_samples):
        windfarm_prod[wf_name].append([])
        for h in range(n_t):
            windfarm_prod[wf_name][i].append(random.randint(0, int(max_capacity)))

# Populate samples for each load
for load_index, load_name in enumerate(loads_samples.keys()):
    max_load = Demand[load_index]
    for i in range(n_samples):
        loads_samples[load_name].append([])
        for h in range(n_t):
            loads_samples[load_name][i].append(random.randint(0, int(max_load)))


model = gb.Model("Unit Commitment")


# production (every generator)
p = model.addVars(n_g, n_t, vtype=gb.GRB.CONTINUOUS)
# s - start-up costs
s = model.addVars(n_g, n_t, vtype=gb.GRB.CONTINUOUS)
# u - on/off status 
u = model.addVars(n_g, n_t, vtype=gb.GRB.BINARY)


for t in range(n_t):
    for g in range(n_g):
        #add constraint min and max production of generators
        model.addConstr(p[g,t] <= P_max.iloc[g].item() *u[g,t])
        model.addConstr(p[g,t] >= P_min.iloc[g].item() *u[g,t])
        if t >= 1:
            #add constraint for up and down ramp (only if t>=1)
            model.addConstr(p[g,t]- p[g,t-1] <= R.iloc[g].item())
            model.addConstr(p[g,t-1]- p[g,t] <= R.iloc[g].item())
            #add constraint for startup cost
            model.addConstr(s[g,t] >= C_st.iloc[g].item()*(u[g,t]-u[g,t-1]))

#add constraints for min on and off time
for t in range(n_t):
    for g in range(n_g):
        if t >= 1:
            min_on_time = min(Up_time.iloc[g].item()+t-1, n_t)
            for tau in range(t, min_on_time):
                model.addConstr(- u[g, t - 1] +u[g, t] -u[g, tau]<=0)

            min_off_time = min(Dw_time.iloc[g].item() +t- 1, n_t)
            for tau in range(t, min_off_time):
                model.addConstr(u[g, t - 1] -u[g, t] +u[g, tau]<=1)


#add netwotk constraints
for t in range(n_t):
    for l in range(n_l):
        expr = sum(Hg[l, g] *p[g,t] for g in range(n_g)) - sum(Hl[l, d] * Demand[d] for d in range(n_d))
        model.addConstr(expr <= F_max.iloc[l].item())
        model.addConstr(expr >= -F_max.iloc[l].item())
        # print("hg:", Hg[l, 1])
        # print("hl:", Hg[l, 1])
        # print("D:", Demand[1])
        
        
# Line technical constraint        
for h in range(n_t):  
    for l, (node1, node2, limit) in enumerate(lines_def):
            
        constraint_lb = model.addConstr(
            -limit,
            gb.GRB.LESS_EQUAL,
            gb.quicksum(
                B[f"Bus{node_i}"][l] * (
                    sum(windfarm_prod[wf][0][h] for wf in WINDFARMS if generator_nodes.get(wf) == node_i) +
                    gb.quicksum(p[(g, h)] for g in P_max if generator_nodes.get(g) == node_i) -
                    sum(loads_samples[load][0][h] for load in data_loads if data_loads.get(load) == node_i) 
                    #gb.quicksum(eta[(n, h)] for n in BUSES if n == f"Bus{node_i}") +
                    #gb.quicksum(delta[(n, h)] for n in BUSES if n == f"Bus{node_i}")
                ) for node_i in range(1, 1 + len(BUSES))
            ),
            name=f"Constraint_lb_line_({node1},{node2})_t={h}"
        )
    
        constraint_ub = model.addConstr(
            limit,
            gb.GRB.GREATER_EQUAL,
            gb.quicksum(
                B[f"Bus{node_i}"][l] * (
                    sum(windfarm_prod[wf][0][h] for wf in WINDFARMS if generator_nodes.get(wf) == node_i) +
                    gb.quicksum(p[(g, h)] for g in P_max if generator_nodes.get(g) == node_i) -
                    sum(loads_samples[load][0][h] for load in data_loads if load_nodes.get(load) == node_i) 
                    #gb.quicksum(eta[(n, h)] for n in BUSES if n == f"Bus{node_i}") +
                    #gb.quicksum(delta[(n, h)] for n in BUSES if n == f"Bus{node_i}")
                ) for node_i in range(1, 1 + len(BUSES))
            ),
            name=f"Constraint_ub_line_({node1},{node2})_t={h}"
        )


#add balancing equation !demand needs to be defined, hourly!
for t in range(n_t):
    model.addConstr(sum(p[g,t] for g in range(n_g))+sum(windfarm_prod[g,t] for g in range(n_w)) == sum(loads_samples[d,t] for d in range(n_d)))


#minimize cost (Opbjective function)
expr = sum(C_op.iloc[g].item() * p[g, t] for g in range(n_g) for t in range(n_t)) + sum(s[g, t] for g in range(n_g) for t in range(n_t))
model.setObjective(sense=gb.GRB.MINIMIZE, expr=expr)

opt = model.optimize()










