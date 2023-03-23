import sys
import networkx as nx
import matplotlib.pyplot as plt
from lp import *

cluster_size = 0  # Should be overwritten by argv[2]
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
        create_solution_from_path_list(graph, path_list)
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


def generate_solution(g, reqs):
    # Choose core using some heuristic
    c: [int] = get_core(g, reqs, cluster_size)
    # Use multi-start dijkstra to create multiple trees
    paths = nx.multi_source_dijkstra_path(g, c)
    create_solution_from_path_list(graph, paths)
    forest = nx.subgraph_view(graph, filter_edge=filter_solution)
    # Connect the core nodes using some approach
    connect_forest(g, reqs, c, forest)
    if not nx.is_connected(forest):
        print('it should be connectd, error')
        exit(1)
    # Return the tree created after the connection
    for u in g.nodes():
        try:
            cycle = nx.find_cycle(forest, u)
            remove_edge_by_cost(graph, cycle, reqs)
        except:
            pass
    return forest


def get_core(g, reqs, core_size):
    o = {}  # o[u] is the sum of requirements
    for u in g.nodes():
        o[u] = sum(req[u][v] for v in g.nodes())
    o = dict(sorted(o.items(), key=lambda item: -item[1]))
    return list(o.keys())[:core_size]


def connect_forest(g, reqs, c, f):
    u = c[0]
    c.remove(u)
    while not len(c) == 0:
        v = c[0]
        best_dist = float('inf')
        best_path = []
        for node in c:
            dist, path = nx.single_source_dijkstra(g, u, target=node)
            if dist < best_dist:
                v = node
                best_dist = dist
                best_path = path
        # insert path in solution
        update_solution_from_path_list(g, best_path)
        # check possible cycles
        try:
            cycle = nx.find_cycle(f, u)
            remove_edge_by_cost(g, cycle, reqs)
        except:
            pass
        # remove from core all nodes connected with u
        to_remove = []
        for node in c:
            if nx.has_path(f, u, node):
                to_remove.append(node)
        for node in to_remove:
            c.remove(node)


def remove_edge_by_cost(g, cycle, reqs):
    big_e = cycle[0]
    big_cost = reqs[big_e[0]][big_e[1]]
    for e in cycle[1:]:
        cost = reqs[e[0]][e[1]]
        if cost > big_cost:
            big_e = e
            big_cost = cost
    g[big_e[0]][big_e[1]]['in_solution'] = False


def calculate_cost(t, r, print_cost=False):
    cost = 0.0
    for i in t.nodes():
        path_list = nx.shortest_path(t, source=i)
        for j in t.nodes():
            if j == i:
                continue
            requirement = max(r[i][j], r[j][i])
            cost += get_path_cost(t, path_list[j]) * requirement
    # Divide the cost by 2
    cost /= 2
    if print_cost:
        print("communication cost:", cost)
    return cost


def filter_solution(n1, n2):
    return graph[n1][n2].get('in_solution', True)


def create_solution_from_path_list(g, path_list):
    for u, v in g.edges():
        g[u][v]['in_solution'] = False

    for path in path_list.values():
        update_solution_from_path_list(g, path)


def update_solution_from_path_list(g, path):
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
                    dummy_node = max(g.number_of_nodes(), curr_biggest + 1)
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
                            g.add_edge(neighbor,
                                       dummy_node,
                                       weight=g[n][neighbor]['weight'],
                                       in_solution=g[n][neighbor]['in_solution'])
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
            if t.has_edge(u, v):
                # if u is dummy of v, remove u
                if nx.get_node_attributes(t, 'father')[u] == v:
                    must_merge = (v, u)  # u will be contracted in v
                # if v is dummy of u, remove v
                elif nx.get_node_attributes(t, 'father')[v] == u:
                    must_merge = (u, v)  # v will be contracted in u
                return c1 + list(set(c2) - set(c1)), must_merge
    return None, None


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


def insert_nodes_bellow(st, node: int, up: int):
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


def print_edges(graph):
    for edge in graph.edges():
        print(edge[0], edge[1])


def main(print_logs=False, plot_tree=False):
    global graph
    d_pos = None
    # To small and some medium size instances, we can change the lines bellow to solution_by_dijkstra()
    # Warning: instead of paths, solution_by_dijkstra() returns the tree
    tree = generate_solution(graph, req)
    root = list(graph.nodes())[0]
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
    print(f'number of clusters: {len(clusters)}')

    if len(clusters) == 1:
        # Generate requirements for subproblem
        # Create vars for MIP
        o_dict = get_o_u(graph, req)
        x_dict, y_dict, f_dict = defining_vars(graph)
        # Generate MIP
        problem = generate_problem(graph, req, o_dict, x_dict, y_dict, f_dict, write_subproblem=True)
        # Calling solver
        solve_problem(problem)
        print("optimal solution: ", value(problem.objective))
        return

    improved = True
    while improved:
        improved = False
        for i in range(len(clusters)):
            for j in range(i, len(clusters)):
                if i == j:
                    continue
                first, second = clusters[i], clusters[j]
                # try to merge the two clusters into one
                merged_cluster, nodes_to_merge = select_two_clusters(graph, tree, first, second)
                if merged_cluster is None:
                    continue
                # create a backup to return to the original graph is needed
                bk_graph = graph.copy()
                # Remove dummies
                if nodes_to_merge is not None:
                    nx.contracted_nodes(graph, nodes_to_merge[0], nodes_to_merge[1], False, False)
                    merged_cluster.remove(nodes_to_merge[1])
                # Create the subtree and the subgraph to work on
                subtree, subgraph = tree.subgraph(merged_cluster), graph.subgraph(merged_cluster)
                # Generate sub-problem requirements
                sp_req = generate_subproblem_req(tree, subtree, req)

                # BEGIN OF DEBUG AREA
                # If is not a tree, stop the algorithm, error
                if not nx.is_tree(subtree):
                    print("subtree created by clusters not valid")
                    if print_logs:
                        for e in subtree.edges():
                            print(e[0], e[1])
                    if plot_tree:
                        nx.draw(subtree, with_labels=True, node_color="tab:red")
                        plt.show()
                    return
                # Plot the subgraph with the tree in red, for analysis
                if plot_tree:
                    d_pos = nx.spring_layout(subgraph)
                    nx.draw_networkx(subgraph, d_pos)
                    nx.draw_networkx(subgraph, d_pos, edgelist=list(subtree.edges()), edge_color='red')
                    plt.show()
                # END OF DEBUG AREA

                # save the current cost of this pair of clusters
                curr_value = calculate_cost(subtree, sp_req)
                # Create vars for MIP
                o_dict = get_o_u(subgraph, sp_req)
                x_dict, y_dict, f_dict = defining_vars(subgraph)
                # Generate MIP
                problem = generate_problem(subgraph, sp_req, o_dict, x_dict, y_dict, f_dict, write_subproblem=True)
                # Calling solver
                solve_problem(problem)
                solution_cost = value(problem.objective)
                print(f'[{i}, {j}]:', LpStatus[problem.status] + ',' + str(solution_cost))

                # Update cost if had a better result
                if curr_value - solution_cost > 3e-11:
                    print(f"updated in: {curr_value - solution_cost}")
                    iterative_cost = iterative_cost - curr_value + solution_cost
                    improved = True
                else:
                    print("Optimized but it wasn't improved")
                    graph = bk_graph
                    tree = nx.subgraph_view(graph, filter_edge=filter_solution)
                    continue

                # Update graph to persist chages
                for e in subgraph.edges():
                    if x_dict[e].varValue == 1.0:
                        graph.edges[e]['in_solution'] = True
                    else:
                        graph.edges[e]['in_solution'] = False
                # Plot the subgraph with the tree in green, for analysis
                if plot_tree:
                    nx.draw_networkx(subgraph, d_pos)
                    nx.draw_networkx(subgraph, d_pos, edgelist=list(subtree.edges()), edge_color='green')
                    plt.show()

                # Redivide the merged cluster in two new
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
    for i in range(len(k) - 1, 0, -1):
        curr_node = k[i]
        if tree.nodes[curr_node]['father'] is not None:
            father = tree.nodes[curr_node]['father']
            print(f'merging {curr_node} in {father}')
            nx.contracted_nodes(graph, father, curr_node, self_loops=False, copy=False)

    print(f'iteration calc: {iterative_cost}')
    calculate_cost(tree, req, True)


if __name__ == '__main__':
    # Reading instance
    n_vertex, n_edges, graph, req = read_instance(sys.argv[1])
    cluster_size = int(sys.argv[2])
    in_cluster = [False for _ in range(n_vertex)]
    main(plot_tree=False, print_logs=False)
