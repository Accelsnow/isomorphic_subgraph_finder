import json
import networkx as nx
import matplotlib.pyplot as plt
import random


def generate_random_graph(num_nodes, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight):
    # Create an undirected graph object
    G = nx.Graph()

    # Generate random weights for nodes and add nodes to the graph
    for i in range(num_nodes):
        node_weight = random.randint(min_node_weight, max_node_weight)
        G.add_node(f"n{i}", weight=node_weight)

    # Create a random spanning tree to ensure connectivity
    tree_edges = nx.random_tree(num_nodes)
    for edge in tree_edges.edges():
        edge_weight = random.randint(min_edge_weight, max_edge_weight)
        G.add_edge(f"n{edge[0]}", f"n{edge[1]}", weight=edge_weight)

    # Generate additional random edges to make the graph non-tree
    while len(G.edges) < num_nodes * 1.5:  # Adjust the multiplier as needed
        i, j = random.sample(range(num_nodes), 2)
        if not G.has_edge(f"n{i}", f"n{j}"):
            edge_weight = random.randint(min_edge_weight, max_edge_weight)
            G.add_edge(f"n{i}", f"n{j}", weight=edge_weight)

    pos = nx.spring_layout(G, iterations=1000, scale=100, k=1 / 3)  # positions for all nodes

    return G, pos


def load_graph_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    G = nx.DiGraph()

    # Add nodes with weights
    for node in data['graph']['nodes']:
        G.add_node(node['id'], weight=node['weight'])

    # Add edges with weights
    for edge in data['graph']['links']:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

    pos = {node: (data['pos'][node][0], data['pos'][node][1]) for node in data['pos']}

    return G, pos


plt.figure(figsize=(15, 15))
# Load graph from JSON file
G, pos = load_graph_from_json('g2.json')

# G, pos = generate_random_graph(20, 5, 10, 1, 5)
# # Dump data into a JSON file
# json_file = "g3.json"
# data = {"graph": nx.node_link_data(G), "pos": {node: [pos[node][0], pos[node][1]] for node in pos}}
# with open(json_file, 'w') as f:
#     json.dump(data, f)

# Visualize the graph

# Draw nodes with labels
node_labels = {node: f"{node}\n{G.nodes[node]['weight']}" for node in G.nodes}
nx.draw_networkx_nodes(G, pos, node_size=1300, node_shape='o', node_color='white', edgecolors='black')
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color='black', font_weight='bold')

# Draw edges
nx.draw_networkx_edges(G, pos, width=2, edge_color='black')

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

# Show the plot
plt.title("Directed Acyclic Graph")
plt.axis('off')
plt.show()
