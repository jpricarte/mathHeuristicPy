from gurobipy import *
from pulp import *
from os import listdir
from os.path import isfile, join

# Reading instance
'''
    read_instance(file_name: str)
    read a instance file and return it values
    :return: (n, m, graph, req), where
    n: int - number of vertices
    m: int - number of edges
    graph: [[int]] - an adjacency matrix
    req: [[float]] - an matrix of requirements (sup trian)
'''
def read_instance(file_name: str) -> (int, int, [[int]], [[float]]):
    with open(file_name, 'r') as instance_file:
        (n, m) = map(lambda x: int(x), instance_file.readline().split())
        graph = [[ -1 for _ in range(n)] for _ in range(n)]
        req = [[ 0.0 for _ in range(n)] for _ in range(n)]
        for _ in range(m):
            # length(u,v) = l
            (u, v, l) = map(lambda x: float(x), instance_file.readline().split())
            u = int(u)
            v = int(v)
            graph[u][v] = l
            graph[v][u] = l
        # main diagonal equals -1
        for i in range(n):
            graph[i][i] = -1
        for i in range(n):
            for j in range(i,n):
                if j == i:
                    req[i][i] = 0.0
                else:
                    r = float(instance_file.readline())
                    req[i][j] = r
                    req[j][i] = 0.0
        
    return (n, m, graph, req)


'''
    get_o_u(req)
    Generate O value for each vertice u
    return: o, an [float]
'''
def get_o_u(req, n) -> [[float]]:
    o = []
    for u in range(n):
        o.append(sum(req[u][v] for v in range(n)))
    return o


'''
    defining_vars(graph: [[int]])
    generate MIP vars
    return: (x, y, f), where:
    x: dict(LPVar): binary x[i][j]
    y: dict(LPVar): binary y[u][i][j]
    f: dict(LPVar): cont   f[u][i][j]
'''
def defining_vars(graph: [[int]], n: int) -> (dict, dict, dict):
    x = LpVariable.dicts("x", [(i,j) for i in range(n) for j in range(i, n) if graph[i][j] >= 0],
                        lowBound=0, upBound=1, cat='Integer')
    y = LpVariable.dicts("y", [(u,i,j) for u in range(n) for i in range(n) for j in range(n) if graph[i][j] >= 0],
                        lowBound=0, upBound=1, cat='Integer')
    f = LpVariable.dicts("f", [(u,i,j) for u in range(n) for i in range(n) for j in range(n) if graph[i][j] >= 0],
                        lowBound=0)
    return (x, y, f)


'''
    defining_objective(graph: [[int]], f: dict)
    Generate objective function
    return: problem, an LpProblem
'''
def defining_objective(graph: [[int]], n: int, f: dict):
    problem = LpProblem("OCT problem", LpMinimize)
    problem += lpSum(graph[i][j] * lpSum((f[u, i, j] + f[u, j, i]) for u in range(n)) 
                        for i in range(n) for j in range(i,n) if graph[i][j] >= 0), "communication_cost"
    return problem


def first_constraint(n, m, graph, req, o, problem, f): # equivalente: 2.13
    for u in range(n):
        problem += lpSum(f[u, u, j] for j in range(n) 
                         if (j != u) and (graph[u][j] >= 0)) == o[u]


def second_constraint(n, m, graph, req, problem, f): # equivalente: 2.12
    for u in range(n):
        for i in range(n):
            if u == i:
                continue
            problem += lpSum(f[u, j, i] - f[u, i, j] for j in range(n) 
                             if graph[i][j] >= 0) == req[u][i]


def third_fourth_constraint(n, m, graph, req, o, problem, f, y): # não tem equivalente
    for u in range(n):
        min_r = min(req[u][i] for i in range(n)) # if req[u][i] > 0)
        M = o[u] # - min_r
        for i in range(n):
            for j in range(i,n):
                if (graph[i][j] < 0): 
                    continue
                problem += f[u,i,j] <= (M * y[u,i,j])
                problem += f[u,j,i] <= (M * y[u,j,i])


def fifth_constraint(n, m, graph, req, problem, y): # não tem equivalente
    for u in range(n):
        problem += lpSum(y[u, i, j] for i in range(n) for j in range(n) 
                         if (j != u) and (j != i) and (graph[i][j] >= 0)) == (n-1)


def sixth_constraint(n, m, graph, req, problem, x, y): # não tem equivalente
    for u in range(n):
        for i in range(n):
            for j in range(i, n):
                if graph[i][j] < 0:
                    continue
                problem += (y[u,i,j] + y[u,j,i]) <= x[i,j]


def seventh_constraint(n, m, graph, req, problem, x): # equivalente: 2.10
    problem += lpSum(x[i,j] for i in range(n) for j in range(i,n) 
                     if (graph[i][j] >= 0)) == (n-1)


def main():
    if len(sys.argv) != 3:
        print('usage: python [path_to]/solver.py <instance> <time_limit_s>')
    filename = sys.argv[1]
    (n, m, graph, req) = read_instance(filename)
    o = get_o_u(req, n)
    (x, y, f) = defining_vars(graph, n)

    problem = defining_objective(graph, n, f)
    first_constraint(n, m, graph, req, o, problem, f)
    second_constraint(n, m, graph, req, problem, f)
    third_fourth_constraint(n, m, graph, req, o, problem, f, y)
    fifth_constraint(n, m, graph, req, problem, y)
    sixth_constraint(n, m, graph, req, problem, x, y)
    seventh_constraint(n, m, graph, req, problem, x)

    solver = GUROBI(timeLimit=int(sys.argv[2]))
    solver.buildSolverModel(problem)
    solver.callSolver(problem)
    solver.findSolutionValues(problem)

    with open('log.csv','a') as logfile:
        line = filename + ',' + LpStatus[problem.status] + ',' + str(value(problem.objective)) + '\n'
        logfile.write(line)


if __name__ == '__main__':
    main()