import time
from maxflow import ford_fulkerson_max_flow
import networkx as nx
import numpy as np
from approximator_max_flow import ApproximatorMaxFlow
from generate import generate_random_graph, create_info
from generate import visualize_graph

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
        A = np.zeros((len(T.edges()), len(G.nodes())))

        C = np.diag([G[u][v]['capacity'] for u, v in T.edges()])
        C_inv = np.linalg.inv(C)  # Compute the inverse of C
        
        for idx, (u, v) in enumerate(T.edges()):
            # Remove edge to create the cut
            T.remove_edge(u, v)
            
            # Find the connected components which represent the two sides of the cut
            components = list(nx.connected_components(T))
            
            # Determine which component each vertex is in
            for i, node in enumerate(G.nodes()):
                if node in components[0]:
                    A[idx, i] = 1
                else:
                    A[idx, i] = 0
            
            # Add the edge back to the tree
            T.add_edge(u, v)
        # Compute R using the formula R = C_inv * A
        R = np.matmul(C_inv, A)
        return R
    
    def approximator_test():
        # Create a sample graph G with random weights and demands
        G = nx.complete_graph(5)
        for u, v in G.edges():
            G[u][v]['capacity'] = np.random.randint(1, 3)
        demands = np.random.randint(-2, 2, size=len(G.nodes()))
        demands[-1] = -np.sum(demands[:-1])  # Adjust the last demand to make the sum zero

        # demands = np.asarray([1, -1, 0, 0, 0])

        print("Demands:", demands)
        for n, demand in zip(G.nodes(), demands):
            G.nodes[n]['demand'] = demand

        # Create the maximum weight spanning tree T from G
        T = nx.maximum_spanning_tree(G, weight='capacity')
        # Create R matrix as per the example
        R = create_R_matrix(T, G)

        # Assume demands are represented by 'demand' attribute on nodes
        demands = np.array([G.nodes[n]['demand'] for n in range(len(G.nodes()))])

        # Instantiate the ApproximatorMaxFlow class and test the flow approximation
        epsilon = 0.1
        info = create_info(G)
        approximator = ApproximatorMaxFlow(G, R, epsilon, info)  # B is not used in this context
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