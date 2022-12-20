import networkx as nx
import matplotlib.pyplot as plt
from lp import *

cluster_size = 10
clusters = []
current_cluster = []
in_cluster = []
clusters_by_node = {}
dummy_nodes = {}
curr_biggest = 0

def read_instance(file_name: str) -> (int, int, nx.Graph, [[float]]):
    g = nx.Graph()
    with open(file_name, 'r') as instance_file:
        (n, m) = map(lambda x: int(x), instance_file.readline().split())
        r = [[0.0 for _ in range(n)] for _ in range(n)]
        for _ in range(m):
            (u, v, l) = map(lambda x: float(x), instance_file.readline().split())
            u, v = int(u), int(v)
            g.add_edge(u, v, weight=l, in_solution=False)
        for i in range(n):
            for j in range(i, n):
                if j == i:
                    r[i][i] = 0.0
                else:
                    requirement = float(instance_file.readline())
                    r[i][j] = requirement
                    r[j][i] = 0.0

    nx.set_node_attributes(g, None, name='father')
    return n, m, g, r


def get_path_cost(g, path):
    path_cost = 0.0
    for i in range(1, len(path)):
        path_cost += g[path[i]][path[i - 1]]['weight']
    return path_cost


def solution_by_dijkstra(g, n, r):
    best_tree = []
    best_cost = float("inf")
    best_root = 0
    for k in range(n):
        cost = 0.0
        path_list = nx.shortest_path(g, source=k)
        update_solution_from_path_list(graph, path_list)
        t = nx.subgraph_view(graph, filter_edge=filter_solution)
        for i in range(n):
            path_list = nx.shortest_path(t, source=i)
            for j in range(n):
                if j == i:
                    continue
                requirement = max(r[0][len(path_list[j]) - 1], r[len(path_list[j]) - 1][0])
                cost += get_path_cost(g, path_list[j]) * requirement
        if cost < best_cost:
            best_tree = t
            best_cost = cost
            best_root = k
    print(best_root, "as root, communication cost:", best_cost)
    return best_tree, best_root, best_cost


def choose_root(g, n, r):
    best_tree = {}
    best_cost = float("inf")
    best_root = 0
    for i in range(n):
        path_list = nx.shortest_path(g, source=i)
        cost = 0.0
        for j in range(n):
            if j == i:
                continue
            requirement = max(r[0][len(path_list[j]) - 1], r[len(path_list[j]) - 1][0])
            cost += get_path_cost(g, path_list[j]) * requirement
        if cost < best_cost:
            best_tree = path_list
            best_cost = cost
            best_root = i
    print(best_root, "as root")
    return best_tree, best_root, best_cost


'''
    calculate_cost(t, r): Calculate cost using the cost definition
    sum_{i in N} sum_{j in N}
'''
def calculate_cost(t, r, print_cost=False):
    cost = 0.0
    for i in t.nodes():
        path_list = nx.shortest_path(t, source=i)
        for j in t.nodes():
            if j == i:
                continue
            requirement = max(r[i][j], r[j][i])
            cost += get_path_cost(t, path_list[j]) * requirement
    if print_cost:
        print("communication cost:", cost)
    return cost


def filter_solution(n1, n2):
    return graph[n1][n2].get('in_solution', True)


def update_solution_from_path_list(g, path_list):
    for u, v in g.edges():
        g[u][v]['in_solution'] = False

    for path in path_list.values():
        for i in range(1, len(path)):
            if len(path) < 2:
                continue
            # Set edge as True
            g[path[i]][path[i - 1]]['in_solution'] = True


def add_to_cluster(u):
    current_cluster.append(u)
    in_cluster[u] = True
    if len(current_cluster) == cluster_size:
        clusters.append([node for node in current_cluster])
        current_cluster.clear()


def divide_tree_bu(t, node, up=None):
    clusters_by_node[node] = [node]
    if len(t[node]) == 1:  # Only one neighbor, is a leaf
        return
    for child in t[node]:  # Generate children's clusters
        if child == up:
            continue
        divide_tree_bu(t, child, node)
    for child in t[node]:  # Merge clusters
        if child == up:
            continue
        if len(clusters_by_node[child]) < cluster_size:
            clusters_by_node[node] += clusters_by_node[child]
        if len(clusters_by_node[node]) >= cluster_size:
            # Deep copy of cluster and create new cluster
            clusters.append([n for n in clusters_by_node[node]])
            for n in clusters_by_node[node]:
                in_cluster[n] = True
            clusters_by_node[node] = [node]
    # Push the final cluster to clusters
    if up is None and clusters_by_node[node] != [node]:
        clusters.append([n for n in clusters_by_node[node]])
        for n in clusters_by_node[node]:
            in_cluster[n] = True


def create_dummies(g: nx.Graph, ns: [int], cs: [[int]], print_logs=False):
    # Create dummies for each node in nodes_list
    is_in_cluster = {n: False for n in ns}
    nodes_list = list(ns.keys())
    for n in nodes_list:
        # Try to find n in each cluster
        for c in cs:
            if n in c:
                if not is_in_cluster[n]:
                    is_in_cluster[n] = True
                else:
                    # Create dummy
                    global curr_biggest
                    dummy_node = max(g.number_of_nodes(), curr_biggest+1)
                    curr_biggest = dummy_node
                    g.add_node(dummy_node, father=n)
                    # For each neighbor of n that's in cluster, change edge
                    neighbors = list(g[n].keys())
                    for neighbor in neighbors:
                        if neighbor in c:
                            if print_logs:
                                print("++++")
                                print(f"creating edge {neighbor} {dummy_node}")
                                print(f"deleting edge {neighbor} {n}")
                            g.add_edge(neighbor, dummy_node, weight=g[n][neighbor]['weight'], in_solution=g[n][neighbor]['in_solution'])
                            g.remove_edge(n, neighbor)
                    # Add a 0-cost edge between the node and its dummy, and changes n to dummy_node in cluster
                    g.add_edge(n, dummy_node, weight=0, in_solution=True)
                    for i in range(len(c)):
                        if c[i] == n:
                            c[i] = dummy_node
                            break


def select_two_clusters(g, t, c1, c2):
    must_merge = None
    for u in c1:
        for v in c2:
            # CAUTION: merging by edge while sharing nodes can break the algorithm
            if t.has_edge(u, v):
                # if u is dummy of v, remove u
                if nx.get_node_attributes(t, 'father')[u] == v:
                    must_merge = (v, u) # u will be contracted in v
                # if v is dummy of u, remove v
                elif nx.get_node_attributes(t, 'father')[v] == u:
                    must_merge = (u, v)  # v will be contracted in u
                return c1 + list(set(c2) - set(c1)), must_merge
    return None, None


'''
    sub-problem_req: map of list of double - A map of aggregated requirements
    requirements: list of list of double - The original requirements matrix
'''


def add_requirements(t, st, base_node, curr_node, prev_node, subproblem_req, requirements):
    list_of_fathers = nx.get_node_attributes(t, 'father')
    if list_of_fathers[base_node]:
        for node in st.nodes():
            subproblem_req[base_node][node] += 0.0
    else:
        for node in st.nodes():
            if list_of_fathers[node] is not None:
                subproblem_req[base_node][node] += 0.0
            elif list_of_fathers[curr_node] is not None:
                subproblem_req[base_node][node] += 0.0
            elif node > base_node:
                subproblem_req[base_node][node] += \
                    max(requirements[curr_node][node], requirements[node][curr_node])
    # Add recursively
    for neighbor in t[curr_node]:
        if neighbor == prev_node:
            continue
        add_requirements(t, st, base_node, neighbor, curr_node, subproblem_req, requirements)


def generate_subproblem_req(t, st, requirements):
    subproblem_requirements = {}
    for u in st.nodes():
        subproblem_requirements[u] = {}
        # If u is a dummy node, all requirements with other nodes must be zero, the original req will be added later
        if nx.get_node_attributes(t, 'father')[u] is not None:
            for v in st.nodes():
                subproblem_requirements[u][v] = 0.0
        # Initialize with original requirements
        else:
            for v in st.nodes():
                if v > u and nx.get_node_attributes(t, 'father')[v] is None:
                    subproblem_requirements[u][v] = requirements[u][v]
                else:
                    subproblem_requirements[u][v] = 0.0
        # Sum requirements from nodes outside the subproblem
        for neighbor in t[u]:
            if neighbor not in st.nodes():
                add_requirements(t, st, u, neighbor, u, subproblem_requirements, requirements)

    return subproblem_requirements


def count_nodes_bellow(st, node, up, counter):
    counter[node] = 1
    for neighbor in st[node]:
        if neighbor == up:
            continue
        counter[node] += count_nodes_bellow(st, neighbor, node, counter)
    return counter[node]


def insert_nodes_bellow(st, node: int, up: int) -> list[int]:
    nodes = []
    for n in st.neighbors(node):
        if n == up:
            continue
        nodes.append(n)
        nodes += insert_nodes_bellow(st, n, node)
    return nodes


def redivide_tree(st):
    bet_cent = nx.betweenness_centrality(st, weight=None, normalized=True, endpoints=True)
    centroid = max(bet_cent, key=bet_cent.get)
    # From here, divide the tree in centroid using knapsack approach
    counter = {}
    count_nodes_bellow(st, centroid, None, counter)
    # Insert in c1 and c2 just the direct neighbors of the centroid
    counter = dict(filter(lambda elem: elem[0] in st.neighbors(centroid), counter.items()))
    c1, c2 = knapsack_approach(centroid, counter, len(st.nodes()))
    # Insert other nodes in each cluster
    bellow_c1 = []
    for n in c1:
        if n == centroid:
            continue
        bellow_c1 += insert_nodes_bellow(st, n, centroid)
    c1 += bellow_c1
    bellow_c2 = []
    for n in c2:
        if n == centroid:
            continue
        bellow_c2 += insert_nodes_bellow(st, n, centroid)
    c2 += bellow_c2
    return c1, c2


def main(print_logs=False, plot_tree=False):
    global graph
    # To small and some medium size instances, we can change the lines bellow to solution_by_dijkstra()
    # Warning: instead of paths, solution_by_dijkstra() returns the tree
    paths, root, _ = choose_root(graph, n_vertex, req)
    update_solution_from_path_list(graph, paths)
    tree = nx.subgraph_view(graph, filter_edge=filter_solution)
    iterative_cost = calculate_cost(tree, req, True)
    if print_logs:
        for e in tree.edges():
            print(e[0], e[1])
    # divide in sub-clusters
    for node in graph.nodes():
        dummy_nodes[node] = set()
    divide_tree_bu(tree, root)
    create_dummies(graph, graph.nodes(), clusters)
    # Creating struct to record values
    last_solution: {(int, int), float} = {}

    improved = True
    # Iteration
    while improved:
        improved = False
        for i in range(len(clusters)):
            for j in range(i, len(clusters)):
                if i == j:
                    continue
                first = clusters[i]
                second = clusters[j]
                merged_cluster, nodes_to_merge = select_two_clusters(graph, tree, first, second)
                if merged_cluster is None:
                    continue
                # Debug
                if print_logs:
                    print("-------------")
                    print(f"{i}: {first}")
                    print(f"{j}: {second}")
                # copy graph to go back if needed
                bk_graph = graph.copy()
                # remove possible dummy if connected by one
                if nodes_to_merge is not None:
                    nx.contracted_nodes(graph, nodes_to_merge[0], nodes_to_merge[1], self_loops=False, copy=False)
                    merged_cluster.remove(nodes_to_merge[1])
                # Subtree, used to calculate things such as subproblem requirements
                subtree = tree.subgraph(merged_cluster)
                # Subgraph, used to generate local solution
                subgraph = graph.subgraph(merged_cluster)
                if not nx.is_tree(subtree):
                    print("subtree created by clusters not valid")
                    if print_logs:
                        for e in subtree.edges():
                            print(e[0], e[1])
                    if plot_tree:
                        nx.draw(subtree, with_labels=True, node_color="tab:red")
                        plt.show()
                    return
                # Generate values for subproblem
                sp_req = generate_subproblem_req(tree, subtree, req)
                o_dict = get_o_u(subgraph, sp_req)
                x_dict, y_dict, f_dict = defining_vars(subgraph)
                # Generate MIP
                problem = generate_problem(subgraph, sp_req, o_dict, x_dict, y_dict, f_dict, write_subproblem=True)
                # Calling solver
                solve_problem(problem)
                solution_cost = value(problem.objective)
                print(LpStatus[problem.status] + ',' + str(solution_cost))
                if (i, j) not in last_solution:
                    last_solution[(i, j)] = calculate_cost(subtree, sp_req)
                if abs(last_solution[(i, j)] - solution_cost) > 1E-5:
                    last_solution[(i, j)] = solution_cost
                    improved = True
                else:
                    print("Optimized but same result was found")
                    graph = bk_graph
                    tree = nx.subgraph_view(graph, filter_edge=filter_solution)
                    continue
                # updating graph
                for e in subgraph.edges():
                    if x_dict[e].varValue == 1.0:
                        graph.edges[e]['in_solution'] = True
                    else:
                        graph.edges[e]['in_solution'] = False
                tree = nx.subgraph_view(graph, filter_edge=filter_solution)
                if plot_tree:
                    nx.draw(subtree, with_labels=True, node_color="tab:green")
                    plt.show()
                c1, c2 = redivide_tree(subtree)
                create_dummies(graph, {c: None for c in merged_cluster}, [c1, c2])

                # Debug only, remove all until the END comment
                merged_cluster = c1 + c2
                subtree = tree.subgraph(merged_cluster)
                if not nx.is_tree(subtree):
                    print("subtree of solution not valid")
                    print(c1)
                    print(c2)
                    if print_logs:
                        for e in tree.edges():
                            print(e[0], e[1])
                    if plot_tree:
                        nx.draw(subtree, with_labels=True, node_color="tab:red")
                        plt.show()
                    return
                if not nx.is_tree(tree.subgraph(c1)):
                    print("error in division")
                if not nx.is_tree(tree.subgraph(c2)):
                    print("error in division")
                # END

                clusters[i] = c1
                clusters[j] = c2

    k = list(graph.nodes().keys())
    k.sort()
    for i in range(len(k)-1, 0, -1):
        curr_node = k[i]
        if tree.nodes[curr_node]['father'] is not None:
            father = tree.nodes[curr_node]['father']
            print(f'merging {curr_node} in {father}')
            nx.contracted_nodes(graph, father, curr_node, self_loops=False, copy=False)

    calculate_cost(tree, req, True)
    print("Iterative cost:", iterative_cost)


if __name__ == '__main__':
    # Reading instance
    n_vertex, n_edges, graph, req = read_instance('./instances/tests/berry35.in')
    in_cluster = [False for _ in range(n_vertex)]
    main(plot_tree=False, print_logs=False)
