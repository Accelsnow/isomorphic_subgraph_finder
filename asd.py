from __future__ import annotations
import itertools
import json
import networkx as nx
import matplotlib.pyplot as plt
import random


def display_graph(graph, position, node_size, dim):
    plt.figure(figsize=(dim, dim))
    node_labels = {node: f"{node}\n{graph.nodes[node]['weight']}" for node in graph.nodes}
    nx.draw_networkx_nodes(graph, position, node_size=node_size, node_shape='o', node_color='white', edgecolors='black')
    nx.draw_networkx_labels(graph, position, labels=node_labels, font_size=12, font_color='black', font_weight='bold')
    nx.draw_networkx_edges(graph, position, width=2, edge_color='black')

    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, position, edge_labels=edge_labels, font_size=12)

    # Show the plot
    plt.title("Network Graph")
    plt.axis('off')
    plt.show()


def generate_and_save_random_network_graph(num_nodes, min_node_weight, max_node_weight, min_edge_weight,
                                           max_edge_weight, file_name):
    """
    Randomly generate network graph with random weights for nodes and edges.
    Node represents a machine. Node weight represents available resource on that machine.
    Edge represents a network link between two machines. Edge weight represents latency of the link.
    The score function depends on machine utilization and network latency.
    Nodes are named as n0, n1, n2, ...
    :param num_nodes:
    :param min_node_weight:
    :param max_node_weight:
    :param min_edge_weight:
    :param max_edge_weight:
    :param file_name: this function automatically dumps generated graph to JSON file with this name
    :return: generated network graph
    """
    dim = 15
    node_size = 2600
    graph = nx.Graph()

    # Generate random weights for nodes and add nodes to the graph
    for i in range(num_nodes):
        node_weight = random.randint(min_node_weight, max_node_weight)
        graph.add_node(f"n{i}", weight=node_weight)

    # Create a random spanning tree to ensure connectivity
    tree_edges = nx.random_tree(num_nodes)
    for edge in tree_edges.edges():
        edge_weight = random.randint(min_edge_weight, max_edge_weight)
        graph.add_edge(f"n{edge[0]}", f"n{edge[1]}", weight=edge_weight)

    # Generate additional random edges to make the graph non-tree
    while len(graph.edges) < num_nodes * 1.5:  # Adjust the multiplier as needed
        i, j = random.sample(range(num_nodes), 2)
        if not graph.has_edge(f"n{i}", f"n{j}"):
            edge_weight = random.randint(min_edge_weight, max_edge_weight)
            graph.add_edge(f"n{i}", f"n{j}", weight=edge_weight)

    position = nx.spring_layout(graph, iterations=1000, scale=1000, k=1 / 2)

    data = {"graph": nx.node_link_data(graph), "dim": dim, "node_size": node_size,
            "pos": {node: [position[node][0], position[node][1]] for node in position}}
    with open(file_name, 'w') as f:
        json.dump(data, f)

    display_graph(graph, position, node_size, dim)
    return graph, position, node_size, dim


def generate_and_save_random_workload_graph(num_nodes, min_node_weight, max_node_weight, file_name):
    """
    Randomly generate workload graph with a single starting node and a single ending node.
    Every path starts with node S and ends with node E. Intermediate nodes are numbered as n0, n1, n2, ...
    Graph is connected and acyclic (although it is undirected, but direction is implied by S->E path)
    There is no edge weight for workload graph.
    :param num_nodes:
    :param min_node_weight:
    :param max_node_weight:
    :param file_name: this function automatically dumps generated graph to JSON file with this name
    :return: generated graph
    """
    dim = 15
    node_size = 2600
    graph = nx.Graph()

    # Generate random weights for nodes and add nodes to the graph
    for i in range(num_nodes):
        node_weight = random.randint(min_node_weight, max_node_weight)
        graph.add_node(f'n{i}', weight=node_weight)

    starting_node = 'S'
    ending_node = 'E'
    graph.add_node(starting_node, weight=0)
    graph.add_node(ending_node, weight=0)

    record = [str(n) for n in graph.nodes() if n != starting_node and n != ending_node]

    # probability based spanning
    while len(record) > 0:
        curr_node = starting_node
        try:
            paths = list(nx.all_shortest_paths(graph, source=starting_node, target=ending_node))
        except nx.NetworkXNoPath:
            paths = []

        if paths:
            path = random.choice(list(paths))
            branch_node = random.choice(path[1:-1])
            curr_node = branch_node

        while len(record) > 0 and (curr_node == starting_node or random.random() < 0.66):
            next_node = record.pop(0)
            graph.add_edge(curr_node, next_node)
            curr_node = next_node

        graph.add_edge(curr_node, ending_node)

    position = nx.spring_layout(graph, iterations=1000, scale=10, k=1 / 2)  # positions for all nodes

    data = {"graph": nx.node_link_data(graph), "dim": dim, "node_size": node_size,
            "pos": {node: [position[node][0], position[node][1]] for node in position}}
    with open(file_name, 'w') as f:
        json.dump(data, f)

    display_graph(graph, position, node_size, dim)
    return graph, position, node_size, dim


def load_graph(json_file):
    """
    Load graph from the given JSON file.
    Note that JSON file must also contain 'dim' and 'node_size' in addition to nx graph dump.
    :param json_file:
    :return: loaded graph
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    dim = int(data["dim"])
    node_size = int(data["node_size"])
    plt.figure(figsize=(dim, dim))
    graph = nx.Graph()

    for node in data['graph']['nodes']:
        graph.add_node(node['id'], weight=node['weight'])

    for edge in data['graph']['links']:
        if "weight" in edge:
            graph.add_edge(edge['source'], edge['target'], weight=edge['weight'])
        else:
            graph.add_edge(edge['source'], edge['target'])

    position = {node: (data['pos'][node][0], data['pos'][node][1]) for node in data['pos']}

    display_graph(graph, position, node_size, dim)
    return graph, position, node_size, dim


def find_isomorphic_subgraph(graph_g: nx.Graph, graph_h: nx.Graph):
    from tqdm import tqdm
    search_space = list(itertools.combinations(graph_g.nodes(), graph_h.number_of_nodes()))
    random.shuffle(search_space)  # is this better?
    for nodes in tqdm(search_space, desc="Searching all sub-graphs"):
        subgraph = graph_g.subgraph(nodes)
        if nx.is_connected(subgraph) and subgraph.number_of_edges() == graph_h.number_of_edges():
            mapping = nx.isomorphism.vf2pp_isomorphism(subgraph, graph_h)
            if mapping is not None:
                success = True
                for ng, nh in mapping.items():
                    if graph_g.nodes[ng]['weight'] < graph_h.nodes[nh]['weight']:
                        success = False
                        break
                if success:
                    return subgraph, mapping
    return None, {}


def display_schedule_graph(graph, position, node_size, dim, subgraph, subgraph_mapping):
    plt.figure(figsize=(dim, dim))

    other_nodes = [node for node in graph.nodes() if node not in subgraph.nodes()]
    node_labels = {node: f"{node}\n{graph.nodes[node]['weight']}" for node in other_nodes}
    for node in subgraph.nodes():
        node_labels[node] = f"{node}/{subgraph_mapping[node]}\n{graph.nodes[node]['weight']}"

    nx.draw_networkx_nodes(graph, position, nodelist=other_nodes, node_size=node_size, node_shape='o',
                           node_color='white', edgecolors='black')
    nx.draw_networkx_nodes(graph, position, nodelist=subgraph.nodes(), node_size=node_size, node_shape='o',
                           node_color='white', edgecolors='red', linewidths=2)
    nx.draw_networkx_labels(graph, position, labels=node_labels, font_size=12, font_color='black', font_weight='bold')

    other_edges = [edge for edge in graph.edges() if edge not in subgraph.edges()]
    nx.draw_networkx_edges(graph, position, edgelist=other_edges, width=2, edge_color='black')
    nx.draw_networkx_edges(graph, position, edgelist=subgraph.edges(), width=4, edge_color='red')

    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, position, edge_labels=edge_labels, font_size=12)

    # Show the plot
    plt.title("Network Graph")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # g = generate_and_save_random_network_graph(20, 5, 10, 1, 5, "grand.json")
    # generate_and_save_random_workload_graph(6, 1, 10, "wrand.json")
    g, gpos, gnode_size, gdim = load_graph('g3.json')
    w, wpos, wnode_size, wdim = load_graph('w3.json')
    load_graph('w1.json')
    load_graph('w2.json')
    match_graph, match_mapping = find_isomorphic_subgraph(g, w)

    if match_graph is None:
        print("No isomorphic subgraph found")
    else:
        print(match_mapping)
        display_schedule_graph(g, gpos, gnode_size, gdim, match_graph, match_mapping)
