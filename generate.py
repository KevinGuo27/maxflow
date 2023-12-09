import matplotlib.pyplot as plt
import networkx as nx
import random

# num_nodes = 1000
# num_edges = 5000
# capacity_range = (1, 100)
def generate_random_graph(num_nodes, num_edges, capacity_range):
    # Create an empty directed graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))

    # Randomly add edges with random capacities
    while G.number_of_edges() < num_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and not G.has_edge(u, v):
            capacity = random.randint(*capacity_range)
            G.add_edge(u, v, capacity=capacity)

    return G


def visualize_graph(G):
    # Choose a layout for our graph - spring layout in this case
    pos = nx.spring_layout(G, scale=2)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color="blue", alpha=0.6)
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)

    # Draw node labels (optional, might be cluttered for large graphs)
    # nx.draw_networkx_labels(G, pos)

    # Show the plot
    plt.show()

def test():
    G = generate_random_graph(1000, 5000, (1, 100))
    visualize_graph(G)
    # Print the edges with capacities
    for u, v, data in G.edges(data=True):
        print(f"Edge ({u}, {v}) has capacity: {data['capacity']}")

if __name__ == "__main__":
    test()