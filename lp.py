from gurobipy import *
from pulp import *

'''
    get_o_u(req)
    Generate O value for each vertice u
    return: o, an [float]
'''


def get_o_u(subgraph, req) -> [[float]]:
    o = {}
    for u in subgraph.nodes():
        o[u] = sum(req[u][v] for v in subgraph.nodes())
    return o


'''
    defining_vars(graph: [[int]])
    generate MIP vars
    return: (x, y, f), where:
    x: dict(LPVar): binary x[i][j]
    y: dict(LPVar): binary y[u][i][j]
    f: dict(LPVar): cont   f[u][i][j]
'''


def defining_vars(subgraph) -> (dict, dict, dict):
    x = LpVariable.dicts("x", [e for e in subgraph.edges()],
                         lowBound=0, upBound=1, cat='Integer')
    y = LpVariable.dicts("y", [(u, i, j) for u in subgraph.nodes() for i, j in subgraph.edges()],
                         lowBound=0, upBound=1, cat='Integer')
    y.update(LpVariable.dicts("y", [(u, j, i) for u in subgraph.nodes() for i, j in subgraph.edges()],
                              lowBound=0, upBound=1, cat='Integer'))
    f = LpVariable.dicts("f", [(u, i, j) for u in subgraph.nodes() for i, j in subgraph.edges()],
                         lowBound=0)
    f.update(LpVariable.dicts("f", [(u, j, i) for u in subgraph.nodes() for i, j in subgraph.edges()],
                              lowBound=0))
    return x, y, f


'''
    defining_objective(graph: [[int]], f: dict)
    Generate objective function
    return: problem, an LpProblem
'''


def defining_objective(subgraph, f: dict):
    problem = LpProblem("OCT_problem", LpMinimize)
    problem += lpSum(subgraph[i][j]['weight'] * lpSum((f[u, i, j] + f[u, j, i]) for u in subgraph.nodes())
                     for i, j in subgraph.edges()), "communication_cost"
    return problem


def first_constraint(subgraph, o, problem, f):  # equivalent: 2.13
    for u in subgraph.nodes():
        problem += lpSum(f[u, u, j] for i, j in subgraph.edges(u)) == o[u], "1st " + str(u)


def second_constraint(subgraph, req, problem, f):  # equivalent: 2.12
    for u in subgraph.nodes():
        for i in subgraph.nodes():
            if u == i:
                continue
            problem += lpSum(f[u, j, i] - f[u, i, j] for _, j in subgraph.edges(i)) \
                       == req[u][i], "2nd_" + str(u) + "-" + str(i)


def third_fourth_constraint(subgraph, req, o, problem, f, y):
    for u in subgraph.nodes():
        min_r = min(req[u][i] for i in subgraph.nodes())
        M = o[u] - min_r
        for i, j in subgraph.edges():
            problem += f[u, i, j] <= (M * y[u, i, j]), "3rd_" + str(u) + "-" + str(i) + "-" + str(j)
            problem += f[u, j, i] <= (M * y[u, j, i]), "4th_" + str(u) + "-" + str(j) + "-" + str(i)


def fifth_constraint(subgraph, problem, y):
    for u in subgraph.nodes():
        problem += lpSum(y[u, i, j] + y[u, j, i] for i, j in subgraph.edges()) \
                   == (len(subgraph.nodes()) - 1), "5th " + str(u)


def sixth_constraint(subgraph, problem, x, y):
    for u in subgraph.nodes():
        for i, j in subgraph.edges():
            problem += (y[u, i, j] + y[u, j, i]) <= x[i, j], "6th_" + str(u) + "-" + str(i) + "-" + str(j)


def seventh_constraint(subgraph, problem, x):  # equivalent: 2.10
    problem += lpSum(x[i, j] for i, j in subgraph.edges()) == (len(subgraph.nodes()) - 1), "7th"


def first_valid_inequality(subgraph, problem, y):
    for u in subgraph.nodes():
        problem += lpSum(y[u, u, j] for _, j in subgraph.edges(u)) >= 1
        for i in subgraph.nodes():
            if u == i:
                continue
            problem += lpSum(y[u, j, i] for _, j in subgraph.edges(i)) == 1


def generate_problem(subgraph, sp_req, o_dict, x_dict, y_dict, f_dict, write_subproblem=False, name="problem.lp"):
    problem = defining_objective(subgraph, f_dict)
    first_constraint(subgraph, o_dict, problem, f_dict)
    second_constraint(subgraph, sp_req, problem, f_dict)
    third_fourth_constraint(subgraph, sp_req, o_dict, problem, f_dict, y_dict)
    fifth_constraint(subgraph, problem, y_dict)
    sixth_constraint(subgraph, problem, x_dict, y_dict)
    seventh_constraint(subgraph, problem, x_dict)
    first_valid_inequality(subgraph, problem, y_dict)
    if write_subproblem:
        problem.writeLP(name)
    return problem


def solve_problem(problem, time_limit=3600, print_msg=False):
    solver = GUROBI(timeLimit=time_limit, msg=print_msg)
    solver.buildSolverModel(problem)
    solver.callSolver(problem)
    solver.findSolutionValues(problem)


'''
    Knapsack approach for cluster splitting
'''


def knapsack_approach(centroid: int, weights, n_nodes: int):  # TODO: MUDAR PARA PD
    c1, c2 = [centroid], [centroid]
    v_dict = LpVariable.dicts("v", [v for v in weights.keys()],
                              lowBound=0, upBound=1, cat='Integer')
    problem = LpProblem("knapsack_problem", LpMaximize)
    problem += lpSum(v_dict[k] * weights[k] for k in weights.keys()), "selected_items"
    problem += lpSum(v_dict[k] * weights[k] for k in weights.keys()) <= (n_nodes // 2), "c1"
    problem.writeLP("knapsack.lp")
    solve_problem(problem)
    for k in v_dict:
        if v_dict[k].varValue == 1.0:
            c1.append(k)
        else:
            c2.append(k)
    return c1, c2
