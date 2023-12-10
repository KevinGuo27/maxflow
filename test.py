import time
from maxflow import ford_fulkerson_max_flow
import networkx as nx
import numpy as np
from approximator_max_flow import ApproximatorMaxFlow
from generate import generate_random_graph, create_info

def timer_function(algorithm, *args, **kwargs):
    """
    Measure the runtime of an algorithm.

    Parameters:
    algorithm (function): The algorithm to be timed.
    *args: Arguments to be passed to the algorithm.
    **kwargs: Keyword arguments to be passed to the algorithm.

    Returns:
    result: The output of the algorithm.
    runtime: The time taken to execute the algorithm.
    """
    start_time = time.time()  # Record the start time
    result = algorithm(*args, **kwargs)  # Execute the algorithm
    end_time = time.time()  # Record the end time

    runtime = end_time - start_time  # Calculate the runtime
    return result, runtime

# Example usage:
# def my_algorithm(params):
#     # Algorithm implementation
#     return result

# result, runtime = timer_function(my_algorithm, params)
# print(f"Runtime: {runtime} seconds")

def test_ford_fulkerson(G):
    source = 'A'  # Replace with your source node
    sink = 'Z'    # Replace with your sink node
    max_flow_value, residual_network = ford_fulkerson_max_flow(G, source, sink)
    print("Maximum flow:", max_flow_value)

def test_route_in_tree():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 4)])
    info = create_info(G)
    b = np.array([-1, 0, 0, 1, -1])
    T = nx.maximum_spanning_tree(G)
    approximator = ApproximatorMaxFlow(G, None, None, info, None)
    f = approximator.route_in_tree(T, b)
    print(f)

def test_approximator():
    def create_R_matrix(T, G):
        # R matrix will have rows corresponding to edges in T and columns for each node in G
        R = np.zeros((len(T.edges()), len(G.nodes())))
        
        for idx, (u, v) in enumerate(T.edges()):
            # Find the cut in G induced by removing edge (u, v) from T
            T.remove_edge(u, v)
            # Find the connected components which represent the two sides of the cut
            components = list(nx.connected_components(T))
            T.add_edge(u, v)  # Add the edge back after finding the cut
            
            # Check which component u and v are in
            u_component = 0 if u in components[0] else 1
            S = components[u_component]  # S is the side of the cut containing u
            S_complement = components[1 - u_component]  # The other side of the cut
            
            # Calculate the cut value as the sum of demands in S minus the sum of demands in S_complement
            b_S = sum(G.nodes[n]['demand'] for n in S)
            c_S = sum(G[u][v]['capacity'] for u, v in G.edges(nbunch=S) if v not in S)
            
            # The entry in R for this edge is the ratio of these sums
            R[idx] = b_S / c_S if c_S != 0 else 0  # Avoid division by zero
        
        return R
    def approximator_test():
        # Create a sample graph G with random weights and demands
        G = nx.complete_graph(5)
        for u, v in G.edges():
            G[u][v]['capacity'] = np.random.randint(1, 100)
        for n in G.nodes():
            G.nodes[n]['demand'] = np.random.randint(-50, 50)

        # Create the maximum weight spanning tree T from G
        T = nx.maximum_spanning_tree(G, weight='capacity')

        # Create R matrix as per the example
        R = create_R_matrix(T, G)

        # Assume demands are represented by 'demand' attribute on nodes
        demands = np.array([G.nodes[n]['demand'] for n in range(len(G.nodes()))])

        # Instantiate the ApproximatorMaxFlow class and test the flow approximation
        epsilon = 0.01
        info = {'num_edges': len(T.edges()), 'num_nodes': len(G.nodes())}
        approximator = ApproximatorMaxFlow(G, R, epsilon, info, None)  # B is not used in this context
        result_flow = approximator(demands)
        
        return result_flow

    # Run the test function
    test_result = approximator_test()
    print("Approximated flow:", test_result)

if __name__ == "__main__":
    # G = nx.DiGraph()
    # G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Z'])
    # G.add_edges_from([('A', 'B', {'capacity': 3}), ('A', 'C', {'capacity': 3}), ('B', 'C', {'capacity': 2}), ('B', 'D', {'capacity': 3}), ('C', 'E', {'capacity': 2}), ('D', 'E', {'capacity': 4}), ('D', 'F', {'capacity': 2}), ('E', 'F', {'capacity': 3}), ('E', 'G', {'capacity': 2}), ('F', 'G', {'capacity': 3}), ('G', 'Z', {'capacity': 3})])
    # test_ford_fulkerson(G)
    test_approximator()