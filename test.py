import time
from maxflow import ford_fulkerson_max_flow

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