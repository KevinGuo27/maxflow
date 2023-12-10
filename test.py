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

if __name__ == "__main__":
    # G = nx.DiGraph()
    # G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Z'])
    # G.add_edges_from([('A', 'B', {'capacity': 3}), ('A', 'C', {'capacity': 3}), ('B', 'C', {'capacity': 2}), ('B', 'D', {'capacity': 3}), ('C', 'E', {'capacity': 2}), ('D', 'E', {'capacity': 4}), ('D', 'F', {'capacity': 2}), ('E', 'F', {'capacity': 3}), ('E', 'G', {'capacity': 2}), ('F', 'G', {'capacity': 3}), ('G', 'Z', {'capacity': 3})])
    # test_ford_fulkerson(G)
    test_route_in_tree()