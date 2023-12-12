import networkx as nx
from networkx import ford_fulkerson_max_flow
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

def ApproximatorMaxFlow(G, R, epsilon):
    """
    A recursive algo that updates softmax of a potential function given in She13
    """
    return

def UltraSparsifyAndReduce(G, kappa):
    return

def CongestionApproximator(G_prime):
    """
    From hierarchial decomposition of RST14

    Methods needed: 
    -   HierarchialDecomp(G): taking in a graph G, return a hierarchial decomposition tree T that has 
        height log(n) and satisfies RST14 guarantees.
        -   PartitionA(S): compute a partition of a cluster S of G into z many subclusters Z_1, ..., Z_z
            such that each subcluster is bounded in size and the set of cutting edges F induces an 1/log^2(n)
            flow
            // This follows from Th'm 3.1
            // Lemma 3.1, Lemma 3.2 implementation
            -   FindSTEdges(A): given an edge set, compute source set, target set, and separation value \eta 
                that 
    -   
    """

    def HierarchialDecomp(G):
        """
        Return a tree data structure
        See RST14 Section 2
        """

        def PartitionA(S):
            """
            Th'm 3.1, Lemma 3.1, Lemma 3.2
            """
        
            def FindSTEdges(A):
                A_s, A_t, eta = None, None, None
                return A_s, A_t, eta
                
            
            return

        return 

    return

def Convert(G, G_prime, RG_prime):
    return



def ford_fulkerson_max_flow(G, source, sink):
    # Run Ford-Fulkerson algorithm
    R = ford_fulkerson_max_flow(G, source, sink)
    # The flow value can be accessed from R.graph['flow_value']
    max_flow_value = R.graph['flow_value']
    return max_flow_value, R