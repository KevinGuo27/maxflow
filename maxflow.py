import networkx as nx
import numpy as np
from math import log, log2

def RecursiveApproxMaxFlow(G, epsilon, b):
    # Step 1: Set κ
    C = 1  # define the constant C as needed
    n = len(G.nodes())
    kappa = C * log(n)**6 * log(log(n))

    # Step 2: UltraSparsifyAndReduce
    G_prime = UltraSparsifyAndReduce(G, kappa)

    # Step 3: CongestionApproximator
    RG_prime = CongestionApproximator(G_prime)

    # Step 4: Convert
    RG = Convert(G, G_prime, RG_prime)

    # Step 5: Return ApproximatorMaxFlow
    return ApproximatorMaxFlow(G, RG, epsilon)

def UltraSparsifyAndReduce(G, kappa):
    return

def CongestionApproximator(G_prime):
    return

def Convert(G, G_prime, RG_prime):
    return

def ApproximatorMaxFlow(G, RG, epsilon):
    return



def ford_fulkerson_max_flow(G, source, sink):
    # Run Ford-Fulkerson algorithm
    R = nx.ford_fulkerson(G, source, sink)
    # The flow value can be accessed from R.graph['flow_value']
    max_flow_value = R.graph['flow_value']
    return max_flow_value, R