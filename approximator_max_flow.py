from typing import Any
import numpy as np
import networkx as nx
from generate import generate_random_graph, create_info
from generate import visualize_graph
class ApproximatorMaxFlow:
    """
    A recursive algo that updates softmax of a potential function given in She13
    """
    def __init__(self, G, R, epsilon, info):
        self.G = G
        self.R = R
        self.epsilon = epsilon
        self.info = info
        self.m = info["num_edges"]
        self.n = info["num_nodes"]
        self.alpha = self.m
        self.B = info["incidence_matrix"]

    def __call__(self, b0):
        b = [b0]
        unscaled_potentials = []
        f, _ , u = self.almostRoute(b0, self.epsilon)
        unscaled_potentials.append(u)
        f = [f]
        T = int(np.log(2 * self.m))

        # Iteratively find the almost routes
        for i in range(1, T + 1):
            # Update b_i based on the flow found in the previous iteration
            b.append(b[i - 1] - self.B.T.dot(f[i - 1]))
            # Find the flow f_i and the potential S_i using almostRoute
            f_i, _ , u = self.almostRoute(b[i], 0.5)  # Using 0.5 as per the instructions
            unscaled_potentials.append(u)
            f.append(f_i)

        # Find the last flow f_T+1 for a flow routing b_T+1 in a maximal spanning tree of G
        b_T_plus_1 = b[i] - self.B.T.dot(f[T])
        f_T_plus_1 = self.route_in_tree(self.info["maximum_spanning_tree"], b_T_plus_1)
        # Sum up all the f_i to get the total flow
        print("flows from AlmostFlow:", f)
        print("Flow from max spanning tree:", f_T_plus_1)
        total_flow = np.sum(f, axis=0) + f_T_plus_1

        # Return the total flow and the last set of potentials
        print("Original demand:", b0)
        print("b_T+1:", b_T_plus_1)
        print("Demand sent by total flow:", self.B.T @ total_flow)
        print("Demand sent by almostflow:", self.B.T @ np.sum(f, axis=0))
        return total_flow, unscaled_potentials
        
    def route_in_tree(self, T, b):
        # Initialize the flow array for all edges in G to zero
        visualize_graph(T)
        f = np.zeros(len(self.G.edges()))

        # Map each edge in G to its index for quick access
        edge_to_index = {edge: idx for idx, edge in enumerate(self.G.edges())}

        # Create a copy of T and b to manipulate without altering the originals
        T_copy = T.copy()
        b_copy = b.copy()

        # Iterate over the leaves and set the flow for the corresponding edges in T
        leaves = set([node for node in T_copy.nodes() if T_copy.degree(node) == 1])
        visited = set()

        while leaves:
            leaf = leaves.pop()
            visited.add(leaf)

            # There is only one neighbor for a leaf in a tree
            neighbors = list(T_copy.neighbors(leaf))
            if(len(neighbors) == 0):
                break
            if len(neighbors) != 1:
                raise ValueError(f"Leaf {leaf} does not have exactly one neighbor")
            neighbor = neighbors[0]

            # Find the edge index in the original graph G
            if (leaf, neighbor) in edge_to_index:
                edge = (leaf, neighbor)
            elif (neighbor, leaf) in edge_to_index:
                edge = (neighbor, leaf)
            else:
                # If the edge isn't in the edge_to_index mapping, it's not part of G
                continue

            # Set the flow on the edge in T to the demand of the leaf
            f[edge_to_index[edge]] = b_copy[leaf]

            # Remove the leaf from T and its demand from the copy of b
            T_copy.remove_node(leaf)
            b_copy[leaf] = 0

            # Check if the neighbor is a leaf after removing the current leaf
            if T_copy.degree(neighbor) == 1 and neighbor not in visited:
                leaves.add(neighbor)

        return f


    def almostRoute(self, b0, epsilon):
        f = np.zeros(self.m)
        unscaled_potentials = []
        print("Initial approximated congestion:", self.R @ b0)
        scaling_factor = (16 * self.epsilon**-1 * np.log2(self.n)) / (2 * self.alpha * np.linalg.norm(self.R @ b0, ord=np.inf))
        b = b0 * scaling_factor
        scale = 1
        C = self.info["capacities"]
        C_inv = np.linalg.inv(C)
        count = 0
        while True:
            potential = self.potentialFunction(f, b)
            while potential < (16 * epsilon**-1 * np.log2(self.n)):
                f *= 17/16
                b *= 17/16
                scale *= 17/16
                potential = self.potentialFunction(f, b)
            

            count += 1
            # print("Iter:", count)
            unscaled_potentials.append((count, potential / scale))
            # Calculate gradient and delta
            grad_potential = self.gradPotentialFunction(f, b)
            delta = np.linalg.norm(C @ grad_potential, ord=1)
            # print("potential:", potential)
            # print("scale:", scale)
            # print("normalized potential:", potential / scale)

            # Update f if delta is large enough
            if delta >= epsilon / 4 and count < self.alpha**2 * epsilon**(-3) * np.log(self.n):
                # Assuming C and Ce are available in the class
                step_size = delta / (1 + 4 * self.alpha**2)
                f -= step_size * np.sign(grad_potential)
                # # f -= step_size * np.diag(C)
                # print("Grad potential:", grad_potential)
                # print("After:", np.sign(grad_potential))
                # print("f:", f)
                # print("b:", b)
            else:
                # Terminate and output f with potentials, undo scaling
                f /= scale  # Undo the scaling of f
                f /= scaling_factor
                break  # Exit the loop

        # Assuming a method to calculate the potentials is defined
        v = self.calculatePotentials(f, b)
        return f, v, unscaled_potentials
    
    def calculatePotentials(self, f, b):
        x2 = 2 * self.alpha * self.R @ (b - self.B.T.dot(f))
        
        # Calculate the gradient of lmax for x2
        p2 = self.grad_lmax(x2)
        
        # Calculate v as the product of R transposed and p2
        # Assuming self.R is a matrix
        v = self.R.T.dot(p2)
        
        return v
    

    def gradPotentialFunction(self, f, b):
        # Compute x1 and its gradient
        C = self.info["capacities"]
        C_inv = np.linalg.inv(C)
        x1 = C_inv @ f
        p1 = self.grad_lmax(x1)

        # Compute x2 and its gradient
        Bf = self.B.T.dot(f)
        x2 = 2 * self.alpha * self.R @ (b - Bf)
        # print("b-Bf:", b - Bf)
        p2 = self.grad_lmax(x2)

        # Compute the gradient of the potential function
        v = self.R.T @ p2
        grad_phi = C_inv @ p1 - 2 * self.alpha * (self.B @ v)

        # print("Gradient of potential function:", grad_phi)
        return grad_phi

    def grad_lmax(self, x):
        # Calculate the sum of exp(xi) and exp(-xi) for all components
        sum_exp = np.sum(np.exp(x) + np.exp(-x), axis=0)
        # Calculate the gradient of lmax
        grad = (np.exp(x) - np.exp(-x)) / sum_exp
        
        return grad

    def lmax(self, x):
        return np.log(np.sum(np.exp(x) + np.exp(-x), axis=0))

    def potentialFunction(self, f, b):
        # Compute the element-wise inverse of C
        C = self.info["capacities"]
        C_inv = np.linalg.inv(C)
        # First term: lmax of the element-wise product of C_inv and f
        term1 = self.lmax(C_inv @ f)

        # Second term: Compute Bf (matrix-vector multiplication)
        Bf = self.B.T.dot(f)
        # Compute 2aR(b - Bf)
        term2_vector = 2 * self.alpha * self.R @ (b - Bf)
        # lmax of the second term
        term2 = self.lmax(term2_vector)

        # print("Term 1:", term1)
        # print("Term 2:", term2)

        # Sum the two terms to get the potential function value
        potential = term1 + term2
        return potential
    
if __name__ == "__main__":
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

    # Create a sample graph G with random weights and demands
    G = nx.complete_graph(5)
    for u, v in G.edges():
        G[u][v]['capacity'] = np.random.randint(1, 10)
    demands = np.random.randint(-2, 2, size=len(G.nodes()))
    demands[-1] = -np.sum(demands[:-1])  # Adjust the last demand to make the sum zero
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
    approx = ApproximatorMaxFlow(G, R, epsilon, info)  # B is not used in this context
    x = np.asarray([100,200,100])
    grad = approx.grad_lmax(x)
    print("Gradient:", grad)