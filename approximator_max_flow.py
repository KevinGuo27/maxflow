from typing import Any
import numpy as np
import networkx as nx
class ApproximatorMaxFlow:
    """
    A recursive algo that updates softmax of a potential function given in She13
    """
    def __init__(self, G, R, epsilon, info, B):
        self.G = G
        self.R = R
        self.epsilon = epsilon
        self.info = info
        self.m = info["num_edges"]
        self.n = info["num_nodes"]
        self.f = None
        self.b = None
        self.alpha = 0.01
        self.B = B

    def __call__(self, b0):
        b = [b0]
        f = [np.zeros_like(b0)]
        T = int(np.log(2 * self.m))

        # Iteratively find the almost routes
        for i in range(1, T + 1):
            # Update b_i based on the flow found in the previous iteration
            b.append(b[i - 1] - self.B.dot(f[i - 1]))
            # Find the flow f_i and the potential S_i using almostRoute
            f_i, _ = self.almostRoute(b[i], 0.5)  # Using 0.5 as per the instructions
            f.append(f_i)

        # Find the last flow f_T+1 for a flow routing b_T+1 in a maximal spanning tree of G
        b_T_plus_1 = b0 - self.B.dot(f[T])
        f_T_plus_1, _ = self.route_in_tree(self.info["maximum_spanning_tree"], b_T_plus_1)

        # Sum up all the f_i to get the total flow
        total_flow = np.sum(f, axis=0) + f_T_plus_1

        # Return the total flow and the last set of potentials
        return total_flow
        
    def route_in_tree(self, T, b):
        f = np.zeros(len(T.edges()))
        # leaves is a numpy array of the leaves of the tree
        leaves = self.info["maximum_spanning_tree_leaves"]

        # Convert T.edges() to a list for indexing purposes
        edges_list = list(T.edges())

        for leaf in leaves:
            # For an undirected graph, there's only one edge for each leaf
            # The edge can be represented as (leaf, neighbor) or (neighbor, leaf)
            neighbors = list(T.neighbors(leaf))
            if len(neighbors) != 1:
                raise ValueError(f"Node {leaf} is not a leaf")

            # Get the edge connected to the leaf
            edge_to_leaf = (leaf, neighbors[0])

            # Find the index of this edge in the list of edges
            if edge_to_leaf in edges_list:
                edge_index = edges_list.index(edge_to_leaf)
            else:
                # If the edge is stored in reverse, (neighbor, leaf)
                edge_to_leaf = (neighbors[0], leaf)
                edge_index = edges_list.index(edge_to_leaf)

            # The flow on this edge is the demand of the leaf
            f[edge_index] = b[leaf]

        return f


    def almostRoute(self, b, epsilon):
        self.f = np.zeros(self.m)
        scaling_factor = (16 * self.epsilon**-1 * np.log(self.n)) / (2 * self.alpha * np.max(self.R @ b))
        self.b = b * scaling_factor
        scale = 1
        while True:
            potential = self.potentialFunction(self.f, self.b)
            while potential < (16 * epsilon**-1 * np.log(self.n)):
                # Scale f and b
                self.f *= 17/16
                self.b *= 17/16
                scale *= 17/16
                potential = self.potentialFunction(self.f, self.b)

            # Calculate gradient and delta
            grad_potential = self.gradPotentialFunction(self.f, self.b)
            delta = np.linalg.norm(grad_potential, 1)

            # Update f if delta is large enough
            if delta >= epsilon / 4:
                # Assuming C and Ce are available in the class
                C_inv = 1 / self.info["capacities"]
                step_size = delta / (1 + 4 * self.alpha**2)
                self.f -= step_size * np.sign(grad_potential) * C_inv
            else:
                # Terminate and output f with potentials, undo scaling
                self.f /= scale  # Undo the scaling of f
                break  # Exit the loop

        # Assuming a method to calculate the potentials is defined
        v = self.calculatePotentials(self.f, self.b)
        return self.f, v
    
    def calculatePotentials(self, f, b):
        x2 = 2 * self.alpha * self.R * (b - self.B.dot(f))
        
        # Calculate the gradient of lmax for x2
        p2 = self.grad_lmax(x2)
        
        # Calculate v as the product of R transposed and p2
        # Assuming self.R is a matrix
        v = self.R.T.dot(p2)
        
        return v
    

    def gradPotentialFunction(self, f, b):
        # Compute x1 and its gradient
        C = self.info["capacities"]
        C_inv = 1 / C
        x1 = C_inv * f
        p1 = self.grad_lmax(x1)

        # Compute x2 and its gradient
        Bf = self.B.dot(f)
        x2 = 2 * self.alpha * self.R @ (b - Bf)
        p2 = self.grad_lmax(x2)

        # Compute the gradient of the potential function
        v = self.R.T @ p2
        grad_phi = C_inv * p1 - 2 * self.alpha * (self.B.T @ v)

        return grad_phi

    def grad_lmax(self, x):
        # Calculate the sum of exp(xi) and exp(-xi) for all components
        sum_exp = np.sum(np.exp(x) + np.exp(-x))
        
        # Calculate the gradient of lmax
        grad = (np.exp(x) - np.exp(-x)) / sum_exp
        
        return grad

    def lmax(self, x):
        return np.log(np.sum(np.exp(x) + np.exp(-x)))

    def potentialFunction(self, f, b):
        # Compute the element-wise inverse of C
        C = self.info["capacities"]
        C_inv = 1 / C
        # First term: lmax of the element-wise product of C_inv and f
        term1 = self.lmax(C_inv * f)

        # Second term: Compute Bf (matrix-vector multiplication)
        Bf = self.B.dot(f)
        # Compute 2aR(b - Bf)
        term2_vector = 2 * self.alpha * self.R * (b - Bf)
        # lmax of the second term
        term2 = self.lmax(term2_vector)

        # Sum the two terms to get the potential function value
        potential = term1 + term2
        return potential