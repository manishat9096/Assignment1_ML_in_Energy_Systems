import gurobipy as gb
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

random.seed(42)
    
def samples_generation(T, n_samples):
    
    windfarm_capacity = pd.read_csv("Capacity of wind farms.csv", delimiter=";")
    loads = pd.read_csv("Loads.csv", delimiter=";")
    
    # Create dictionaries to store data samples
    loads_samples = {f"L{i+1}": [] for i in range(len(loads))}
    windfarm_samples = {f"WF{i+1}": [] for i in range(len(windfarm_capacity))}
    
    # Populate samples for each wind farm
    for wf_index, wf_name in enumerate(windfarm_samples.keys()):
        max_capacity = windfarm_capacity['wind'][wf_index]
        for i in range(n_samples):
            windfarm_samples[wf_name].append([])
            for h in range(T):
                windfarm_samples[wf_name][i].append(random.randint(0, int(max_capacity)))
    
    # Populate samples for each load
    for load_index, load_name in enumerate(loads_samples.keys()):
        max_load = loads['load'][load_index]
        for i in range(n_samples):
            loads_samples[load_name].append([])
            for h in range(T):
                loads_samples[load_name][i].append(random.randint(0, int(max_load)))
                
    return windfarm_samples, loads_samples


def opti_schedule(T, windfarm_samples, loads_samples, sample_n):
    
    # Define ranges
    GENERATORS_ALL = ['G1', 'G2', 'G3', 'WF1', 'WF2']
    GENERATORS = ['G1', 'G2', 'G3']
    WINDFARMS =  ['WF1', 'WF2'] 
    LOADS = ['L1', 'L2', 'L3']
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
    
    # Parameter
    M = 10**9
    
    # Import data
    B = pd.read_csv("B (power transfer factor of each bus to each line).csv", delimiter=";")
    lines_capacity = pd.read_csv("Transmission capacity of lines.csv", delimiter=";") 
        
    P_max_data = pd.read_csv("Maximum production of generating units.csv", delimiter=";") 
    P_min_data = pd.read_csv("Minimum production of generating units.csv", delimiter=";") 
    R_rate_data = pd.read_csv("Ramping rate of generating units.csv", delimiter=";") 
    DT_min_data = pd.read_csv("Minimum down time of generating units.csv", delimiter=";") 
    UT_min_data = pd.read_csv("Minimum up time of generating units.csv", delimiter=";") 
    
    P_cost_data = pd.read_csv("Production cost of generating units.csv", delimiter=";") 
    start_cost_data = pd.read_csv("Start-up cost of generating units.csv", delimiter=";") 
    
    P_max = {}
    P_min = {}
    R_rate = {}
    DT_min = {}
    UT_min = {}
    P_cost = {}
    start_cost = {}
    
    for unit_index, unit in enumerate(GENERATORS):    
        P_max[unit] = P_max_data['pgmax'][unit_index]
        P_min[unit] = P_min_data['pgmin'][unit_index]
        R_rate[unit] = R_rate_data['ru'][unit_index]
        DT_min[unit] = DT_min_data['ld'][unit_index]
        UT_min[unit] = UT_min_data['lu'][unit_index]
        P_cost[unit] = P_cost_data['cost_op'][unit_index]
        start_cost[unit] = start_cost_data['cost_st'][unit_index]
    
    # Line definiton
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
        lines_def[line] = tuple(list(lines_def[line]) + 
                                 [lines_capacity['fmax'][line]])
    
    # Define gurobi model
    model = gb.Model("optimization_model")
    
    # Set time limit
    model.Params.TimeLimit = 100
    
    # Add variables
    generator_production = {
        (g, h): model.addVar(
            name=f'Electricity production of generator {g} t={h}'
        ) for g in GENERATORS for h in range(T)
    }
    
    generator_start_cost = {
        (g, h): model.addVar(
            lb = 0,
            name=f'Starting cost of generator {g} t={h}'
        ) for g in GENERATORS for h in range(T)
    }
    
    generator_commitment = {
        (g, h): model.addVar(
            vtype=gb.GRB.BINARY,
            name=f'Committing generator {g} t={h}'
        ) for g in GENERATORS for h in range(T)
    }
    
    # Slack variables
    eta = {
        (n, h): model.addVar(
            lb = 0,
            name=f'Slack variable eta {n} t={h}'
        ) for n in BUSES for h in range(T)
    }
    
    delta = {
        (n, h): model.addVar(
            lb = 0,
            name=f'Slack variable delta {n} t={h}'
        ) for n in BUSES for h in range(T)
    }
    
    
    # Set objective function
    ED_objective = gb.quicksum(P_cost[g] * generator_production[(g, h)] + generator_start_cost[(g, h)] 
                               for g in GENERATORS for h in range(T)) + M * (gb.quicksum(eta[(n, h)] + delta[(n, h)] for n in BUSES for h in range(T)))
    model.setObjective(ED_objective, gb.GRB.MINIMIZE)
    
    # Add constraints
    for h in range(T):
        
        # Generation technical limits
        constr_gen_min = {
            model.addConstr(
                generator_commitment[(g,h)] * P_min[g],
                gb.GRB.LESS_EQUAL,
                generator_production[(g,h)],
                name=f'Min generation limit of generator {g} t={h}',
            )
            for g in GENERATORS
        }
        
        constr_gen_max = {
            model.addConstr(
                generator_production[(g,h)],
                gb.GRB.LESS_EQUAL,
                generator_commitment[(g,h)] * P_max[g],
                name=f'Max generation limit of generator {g} t={h}',
            )
            for g in GENERATORS
        }
        
        # Ramping-up/down constraint 
        if h > 0 :
            ramp_up = {
                model.addConstr(
                    generator_production[(g,h)] - generator_production[(g,h-1)],
                    gb.GRB.LESS_EQUAL,
                    R_rate[g],
                    name=f'Ramping up rate of generator {g} t={h}',
                )
                for g in GENERATORS
            }
            
            ramp_down = {
                model.addConstr(
                    generator_production[(g,h-1)] - generator_production[(g,h)],
                    gb.GRB.LESS_EQUAL,
                    R_rate[g],
                    name=f'Ramping down rate of generator {g} t={h}',
                )
                for g in GENERATORS
            }
            
        # Balance equation
        balance_constraint = model.addConstr(
            sum(windfarm_samples[wf][0][h] for wf in WINDFARMS) +  gb.quicksum(generator_production[(g, h)] for g in GENERATORS),
            gb.GRB.EQUAL,
            sum(loads_samples[load][0][h] for load in LOADS) + gb.quicksum(eta[(n,h)] - delta[(n,h)] for n in BUSES),
            name=f'Balance equation t={h}'
        )
        
        # Line technical constraint        
        for l, (node1, node2, limit) in enumerate(lines_def):
                
            constraint_lb = model.addConstr(
                -limit,
                gb.GRB.LESS_EQUAL,
                gb.quicksum(
                    B[f"Bus{node_i}"][l] * (
                        sum(windfarm_samples[wf][sample_n][h] for wf in WINDFARMS if generator_nodes.get(wf) == node_i) +
                        gb.quicksum(generator_production[(g, h)] for g in GENERATORS if generator_nodes.get(g) == node_i) -
                        sum(loads_samples[load_][sample_n][h] for load_ in LOADS if load_nodes.get(load_) == node_i) -
                        gb.quicksum(eta[(n, h)] for n in BUSES if n == f"Bus{node_i}") +
                        gb.quicksum(delta[(n, h)] for n in BUSES if n == f"Bus{node_i}")
                    ) for node_i in range(1, 1 + len(BUSES))
                ),
                name=f"Constraint_lb_line_({node1},{node2})_t={h}"
            )
    
            constraint_ub = model.addConstr(
                limit,
                gb.GRB.GREATER_EQUAL,
                gb.quicksum(
                    B[f"Bus{node_i}"][l] * (
                        sum(windfarm_samples[wf][sample_n][h] for wf in WINDFARMS if generator_nodes.get(wf) == node_i) +
                        gb.quicksum(generator_production[(g, h)] for g in GENERATORS if generator_nodes.get(g) == node_i) -
                        sum(loads_samples[load_][sample_n][h] for load_ in LOADS if load_nodes.get(load_) == node_i) -
                        gb.quicksum(eta[(n, h)] for n in BUSES if n == f"Bus{node_i}") +
                        gb.quicksum(delta[(n, h)] for n in BUSES if n == f"Bus{node_i}")
                    ) for node_i in range(1, 1 + len(BUSES))
                ),
                name=f"Constraint_ub_line_({node1},{node2})_t={h}"
            )
            
        # Start-up cost constraint
        if h > 0 :
            start_up_constraint = {
                model.addConstr(
                    start_cost[g] * (generator_commitment[(g,h)] - generator_commitment[(g,h-1)]),
                    gb.GRB.LESS_EQUAL,
                    generator_start_cost[(g,h)],
                    name=f'Start-up cost of generator {g} t={h}',
                )
                for g in GENERATORS
            }
            
            
            for g in GENERATORS : 
                min_DT = min(T, h + DT_min[g])
                minimum_DT_constr = {
                    model.addConstr(
                        - generator_commitment[(g,h-1)] + generator_commitment[(g,h)] - generator_commitment[(g,tau)],
                        gb.GRB.LESS_EQUAL,
                        0,
                        name=f'Minimum DT of generator {g} t={h} tau={tau}',
                    )
                    for tau in range(h, min_DT)
                }
                
                min_UT = min(T, h + DT_min[g])
                minimum_UT_constr = {
                    model.addConstr(
                        generator_commitment[(g,h-1)] - generator_commitment[(g,h)] + generator_commitment[(g,tau)],
                        gb.GRB.LESS_EQUAL,
                        1,
                        name=f'Minimum UT of generator {g} t={h} tau={tau}',
                    )
                    for tau in range(h, min_UT)
                }
    

    model.optimize()
    print("Status", model.status) 
    print("Objective function", model.ObjVal)
    
    # Plotting Results
    production_data = {g: [generator_production[(g, h)].X for h in range(T)] for g in GENERATORS}
    unit_commitment = {g: [generator_commitment[(g, h)].X for h in range(T)] for g in GENERATORS}
    
    eta_data = {n: [eta[(n, h)].X for h in range(T)] for n in BUSES}
    delta_data = {n: [delta[(n, h)].X for h in range(T)] for n in BUSES}
    
    load_data = {l: loads_samples[l][sample_n] for l in LOADS}
    wf_data = {wf: windfarm_samples[wf][sample_n] for wf in WINDFARMS}
    
    # Plot Generator Production for each generator
    plt.figure(figsize=(10, 6), dpi=300)
    for g, production in production_data.items():
        plt.plot(range(T), production, label=f'Production of {g}')
    plt.xlabel('Time (hours)', fontsize=14)
    plt.ylabel('Electricity Production (MW)', fontsize=14)
    plt.title('Generator Production over Time', fontsize=16)
    plt.xlim(0, T-1)
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plot Generator Commitment for each generator  
    fig, axes = plt.subplots(len(unit_commitment), 1, figsize=(16, 3 * len(unit_commitment)), sharex=True, dpi=300)
    for ax, (g, commitment) in zip(axes, unit_commitment.items()):
        ax.bar(range(T), commitment, color='skyblue', label=f'Commitment of {g}')
        ax.set_ylabel('Commitment', fontsize=14)
        ax.set_title(f'Generator Commitment over Time for {g}', fontsize=16)
        ax.grid(axis='y')
        ax.legend()
    plt.xlim(-0.5, T-0.5)
    plt.xlabel('Time (hours)', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Plot Slack Variables Values for each bus
    plt.figure(figsize=(10, 6), dpi=300)
    for n, eta_vals in eta_data.items():
        if any(val != 0 for val in eta_vals):  # Check if any value is non-zero
            plt.plot(range(T), eta_vals, label=f'Eta of {n}', linestyle='--')
    for n, delta_vals in delta_data.items():
        if any(val != 0 for val in delta_vals):  # Check if any value is non-zero
            plt.plot(range(T), [-val for val in delta_vals], label=f'Delta of {n}', linestyle='-.')
    plt.xlabel('Time (hours)', fontsize=14)
    plt.ylabel('Slack Variables Values', fontsize=14)
    plt.title('Slack Variables Values over Time', fontsize=16)
    plt.xlim(0, T-1)
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plot Load data for each load
    plt.figure(figsize=(10, 6), dpi=300)
    for l, load_vals in load_data.items():
        plt.plot(range(T), load_vals, label=f'Load of {l}', linestyle=':')
    total_load = [sum(load_data[l][h] for l in LOADS) for h in range(T)]
    plt.plot(range(T), total_load, label='Total Load', linestyle='-', color='red')
    plt.xlabel('Time (hours)', fontsize=14)
    plt.ylabel('Load (MW)', fontsize=14)
    plt.title('Load over Time', fontsize=16)
    plt.xlim(0, T-1)
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plot Wind Farm production for each wind farm
    plt.figure(figsize=(10, 6), dpi=300)
    for wf, wf_vals in wf_data.items():
        plt.plot(range(T), wf_vals, label=f'WF Production of {wf}', linestyle='--')
    total_prod = [sum(wf_data[wf][h] for wf in WINDFARMS) for h in range(T)]
    plt.plot(range(T), total_prod, label='Total WF Production', linestyle='-', color='red')
    plt.xlabel('Time (hours)', fontsize=14)
    plt.ylabel('WF Production (MW)', fontsize=14)
    plt.title('Wind Farm Production over Time', fontsize=16)
    plt.xlim(0, T-1)
    plt.legend()
    plt.grid()
    plt.show()

    return unit_commitment

### Step 2
## Solving
T = 24
n_samples = 5
windfarm_samples, loads_samples = samples_generation(T, n_samples)
res_commitment = []            
            
for sample in range(n_samples):
    res = opti_schedule(T, windfarm_samples, loads_samples, sample)
    res_commitment.append(res)
    
    
## Creating Dataset
wf1_values = [val for sublist in windfarm_samples['WF1'] for val in sublist]
wf2_values = [val for sublist in windfarm_samples['WF2'] for val in sublist]

L1_values = [val for sublist in loads_samples['L1'] for val in sublist]
L2_values = [val for sublist in loads_samples['L2'] for val in sublist]
L3_values = [val for sublist in loads_samples['L3'] for val in sublist]


X = pd.DataFrame({'WF1': wf1_values, 'WF2': wf2_values,
                  'L1': L1_values, 'L2': L2_values, 'L3': L3_values})

y_G1_values = []
y_G2_values = []
y_G3_values = []

for s in range(len(res_commitment)):
    y_G1_values += res_commitment[s]["G1"]
    y_G2_values += res_commitment[s]["G2"]
    y_G3_values += res_commitment[s]["G3"]




