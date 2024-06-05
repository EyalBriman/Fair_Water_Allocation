import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
from sklearn.decomposition import PCA
gp.setParam('OutputFlag', 0)
from scipy.stats import entropy

def Util(demands, supply,storage,E,C):
    num_agents, num_time_steps = demands.shape
    model = gp.Model("Utilitarian")
    w = model.addVars(num_agents, num_time_steps, name="w", lb=0, ub=GRB.INFINITY)
    alpha = model.addVars(num_agents, name="alpha", lb=0, ub=1)
    model.setObjective(alpha.sum(), GRB.MAXIMIZE)
    for i in range(num_agents):
        for t in range(num_time_steps):
            model.addConstr(w[i, t] >= alpha[i] * demands[i, t], name=f"tightness_{i}_{t}")
    if storage==0:
        for t in range(num_time_steps):
            model.addConstr(gp.quicksum(w[i, t] for i in range(num_agents)) <= supply[t], f"supply_constraint_{t}")
    elif storage==1:
        infint_cap(model, w, supply, num_time_steps, E)
    else:
        finit_cap(model, w, supply, num_time_steps, E,C)
    model.optimize()
    if model.status == GRB.OPTIMAL:
        obj_value = model.objVal
        alpha_values = {i: alpha[i].X for i in range(num_agents)}
        return obj_value, alpha_values
    else:
        print("Optimization did not converge.")
        return None, None, None

def Egal(demands, supply,storage,E,C):
    num_agents, num_time_steps = demands.shape
    model = gp.Model("Egalitarian")
    w = model.addVars(num_agents, num_time_steps, name="w", lb=0, ub=GRB.INFINITY)
    alpha = model.addVars(num_agents, name="alpha", lb=0, ub=1)
    alpha_min = model.addVar(name="alpha_min", lb=0, ub=1)
    model.setObjective(alpha_min, GRB.MAXIMIZE)
    for i in range(num_agents):
        for t in range(num_time_steps):
            model.addConstr(w[i, t] >= alpha[i] * demands[i, t], name=f"tightness_{i}_{t}")
    for i in range(num_agents):
        model.addConstr(alpha_min <= alpha[i], name=f"alpha_consistency_{i}")
    if storage==0:
        for t in range(num_time_steps):
            model.addConstr(gp.quicksum(w[i, t] for i in range(num_agents)) <= supply[t], f"supply_constraint_{t}")
    elif storage==1:
        infint_cap(model, w, supply, num_time_steps, E)
    else:
        finit_cap(model, w, supply, num_time_steps, E,C)
    model.optimize()
    if model.status == GRB.OPTIMAL:
        obj_value = model.objVal
        alpha_values = {i: alpha[i].X for i in range(num_agents)}
        return obj_value, alpha_values
    else:
        print("Optimization did not converge.")
        return None, None

def Nash(demands, supply,storage,E,C):
    model = gp.Model("Nash")
    num_agents, num_time_steps = demands.shape
    alpha = {}
    alpha_log = {}
    w = {}
    for i in range(num_agents):
        alpha[i] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"alpha_{i}")
        alpha_log[i] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"alpha_log_{i}")
        for t in range(num_time_steps):
            w[i, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"w_{i}_{t}")
    for i in range(num_agents):
        model.addGenConstrLog(alpha[i], alpha_log[i])
    model.setObjective(gp.quicksum(alpha_log[i] for i in range(num_agents)), GRB.MAXIMIZE)
    for i in range(num_agents):
        for t in range(num_time_steps):
            model.addConstr(w[i, t] >= alpha[i] * demands[i, t], f"tightness_constraint_{i}_{t}")
    if storage==0:
        for t in range(num_time_steps):
            model.addConstr(gp.quicksum(w[i, t] for i in range(num_agents)) <= supply[t], f"supply_constraint_{t}")
    elif storage==1:
        infint_cap(model, w, supply, num_time_steps, E)
    else:
        finit_cap(model, w, supply, num_time_steps, E,C)
    model.optimize()
    if model.status == GRB.OPTIMAL:
        obj_value = model.objVal
        alpha_values = {i: alpha[i].X for i in range(num_agents)}
        return obj_value, alpha_values
    else:
        print("Optimization did not converge.")
        return None, None

def generate_demands(num_agents, num_time_step):
    demands = np.random.randint(1, 101, size=(num_agents, num_time_steps))
    demands_sum_per_agent = np.sum(demands, axis=1)
    target_sum_per_agent = np.mean(demands_sum_per_agent)
    for i in range(num_agents):
        adjustment_factor = target_sum_per_agent / demands_sum_per_agent[i]
        demands[i, :] = np.round(demands[i, :] * adjustment_factor).astype(int)
    row_sums = np.sum(demands, axis=1)
    max_sum = np.max(row_sums)
    min_sum = np.min(row_sums)
    for i in range(num_agents):
        while row_sums[i] < max_sum:
            min_index = np.argmin(demands[i, :])
            demands[i, min_index] += 1
            row_sums[i] += 1
    return demands


def infint_cap(model, w, supply, num_time_steps, E):
    X = model.addVars(num_time_steps, name="X", lb=0, ub=GRB.INFINITY)
    model.addConstr(gp.quicksum(w[i, 0] for i in range(num_agents)) <= supply[0], name="supply_constraint_0")
    for t in range(1, num_time_steps):
        model.addConstr(gp.quicksum(w[i, t] for i in range(num_agents)) <= supply[t] + X[t], name=f"supply_constraint_{t}")
        model.addConstr(X[t] <= (X[t - 1] + supply[t - 1] - gp.quicksum(w[i, t - 1] for i in range(num_agents))) * E[t], name=f"storage_constraint_{t}")
    
    
def finit_cap(model, w, supply, num_time_steps, E,C):
    X = model.addVars(num_time_steps, name="X", lb=0, ub=C)
    model.addConstr(gp.quicksum(w[i, 0] for i in range(num_agents)) <= supply[0], name="supply_constraint_0")
    for t in range(1, num_time_steps):
        model.addConstr(gp.quicksum(w[i, t] for i in range(num_agents)) <= supply[t] + X[t], name=f"supply_constraint_{t}")
        model.addConstr(X[t] <= (X[t - 1] + supply[t - 1] - gp.quicksum(w[i, t - 1] for i in range(num_agents))) * E[t], name=f"storage_constraint_{t}")
    for t in range(num_time_steps):
        model.addConstr(X[t] <= C, name=f"upper_limit_constraint_{t}")
        
def calculate_alpha(demands1, supply, storage, E, C):
    num_agents, num_time_steps = demands1.shape
    alphas = []
    supply = supply / num_agents
    for tt in range(num_agents):
        demands = demands1[tt]
        model = gp.Model("No")
        w = model.addVars(1, num_time_steps, name="w", lb=0, ub=GRB.INFINITY)
        alpha = model.addVar(name="alpha", lb=0, ub=1)
        model.setObjective(alpha, GRB.MAXIMIZE)
        for t in range(num_time_steps):
            model.addConstr(w[0,t] >= alpha * demands[t], name=f"tightness_{t}")
        if storage == 1:
            infint_cap1(model, w, supply, num_time_steps, E)
        else:
            finit_cap1(model, w, supply, num_time_steps, E, C/num_agents)
        model.optimize()
        if model.status == GRB.OPTIMAL:
            alphas.append(model.objVal)
        else:
            print("Optimization did not converge.")
            return None, None, None

    return (
        np.mean(alphas) if alphas else None,
        np.min(alphas) if alphas else None,
        np.max(alphas) if alphas else None
    )



def DRF(demands, supply, storage, E, C):
    num_agents, num_time_steps = demands.shape
    model = gp.Model("DRF")

    # Decision variables
    x = model.addVar(name="x", lb=0, ub=GRB.INFINITY)  # Maximum allocation of dominant resources
    a = model.addVars(num_agents, name="a", lb=0, ub=GRB.INFINITY)  # Allocation of dominant resource for each user

    # Objective function: maximize x
    model.setObjective(x, GRB.MAXIMIZE)

    # Constraints: allocation of dominant resource for each user >= x * demand
    for i in range(num_agents):
        for t in range(num_time_steps):
            model.addConstr(a[i] >= x * demands[i, t], name=f"allocation_{i}_{t}")

    # Supply constraint: total allocation of dominant resource <= supply
    model.addConstr(gp.quicksum(a[i] for i in range(num_agents)) <= supply, name="supply_constraint")

    # Adjust for storage and E, C if necessary
    if storage == 1:
        infint_cap(model, a, supply, num_time_steps, E)
    elif storage == 2:
        finit_cap(model, a, supply, num_time_steps, E, C)
    model.optimize()
    if model.status == GRB.OPTIMAL:
        max_allocation = x.X
        allocation = {i: a[i].X for i in range(num_agents)}
        return max_allocation, allocation
    else:
        print("Optimization did not converge.")
        return None, None
    


def infint_cap1(model, w, supply, num_time_steps, E):
    X = model.addVars(num_time_steps, name="X", lb=0, ub=GRB.INFINITY)
    model.addConstr(w[0,0] <= supply[0], name="supply_constraint_0")
    for t in range(1, num_time_steps):
        model.addConstr(w[0,t] <= supply[t] + X[t], name=f"supply_constraint_{t}")
        model.addConstr(X[t] <= (X[t - 1] + supply[t - 1] - w[0,t - 1]) * E[t], name=f"storage_constraint_{t}")


    
def finit_cap1(model, w, supply, num_time_steps, E, C):
    X = model.addVars(num_time_steps, name="X", lb=0, ub=C)
    model.addConstr(w[0,0] <= supply[0], name="supply_constraint_0")
    for t in range(1, num_time_steps):
        model.addConstr(w[0,t] <= supply[t] + X[t], name=f"supply_constraint_{t}")
        model.addConstr(X[t] <= (X[t - 1] + supply[t - 1] - w[0,t - 1]) * E[t], name=f"storage_constraint_{t}")
    for t in range(num_time_steps):
        model.addConstr(X[t] <= C, name=f"upper_limit_constraint_{t}")


def generate_supply(num_time_steps,num_agents):
    agent=random.randint(2, num_agents)
    supply_per_time_step = np.random.randint(1,20*agent, size=num_time_steps)
    supply_per_time_step[0]=40*agent
    return supply_per_time_step


    
val_fair_nash={}
val_fair_egal={}
val_fair_util={}
val_envy_util={}
val_envy_nash={}
val_envy_egal={}
val_fair_no={}
val_envy_no={}
num_time_steps = 12
for storage in [0,50,100,200,1]:
    key=(storage)
    val_fair_nash[key]=[]
    val_fair_egal[key]=[]
    val_fair_util[key]=[]
    val_envy_util[key]=[]
    val_envy_nash[key]=[]
    val_envy_egal[key]=[]
    val_envy_no[key]=[]
    val_fair_no[key]=[]
    for num_agents in [100]:
        for rep in range(500):
            E = np.linspace(0.9, 1.0, 12)
            C=storage*num_agents
            demands= np.random.dirichlet([1]*num_time_steps, size=num_agents)*1000+1
            supply=np.random.dirichlet([1]*num_time_steps)*random.randint(500, 1000)*num_agents+1
            total_demands = np.sum(demands, axis=0)
            ratios = supply / total_demands
            min_ratio = np.min(ratios)
            kl_divergences = []
            for i in range(demands.shape[0]): 
                for j in range(i + 1, demands.shape[0]):
                    kl_divergences.append(np.sum(np.abs(demands[i] - demands[j])))
            average_kl_divergence = np.mean(kl_divergences)
            obj_util, alpha_values_util = Util(demands, supply,storage,E,C)
            obj_egal, alpha_values_egal = Egal(demands, supply,storage,E,C)
            obj_nash, alpha_values_nash = Nash(demands, supply,storage,E,C)
            mean_alpha, min_alpha, max_alpha=calculate_alpha(demands, supply, storage, E, C)
            val_fair_nash[key].append([np.mean(list(alpha_values_nash.values())),average_kl_divergence,min_ratio])
            val_fair_egal[key].append([np.mean(list(alpha_values_egal.values())),average_kl_divergence,min_ratio])
            val_fair_util[key].append([np.mean(list(alpha_values_util.values())),average_kl_divergence,min_ratio])
            val_fair_no[key].append([mean_alpha,average_kl_divergence,min_ratio])
            val_envy_util[key].append([min(alpha_values_util.values())/max(alpha_values_util.values()),average_kl_divergence,min_ratio])
            val_envy_nash[key].append([min(alpha_values_egal.values())/max(alpha_values_egal.values()),average_kl_divergence,min_ratio])
            val_envy_egal[key].append([min(alpha_values_nash.values())/max(alpha_values_nash.values()),average_kl_divergence,min_ratio])    
            val_envy_no[key].append([min_alpha/max_alpha,average_kl_divergence,min_ratio])
