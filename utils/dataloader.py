import networkx as nx
import numpy as np


def load_graph(dataset):
    """Loads a graph dataset as networkx graph object."""
    G = nx.Graph()
    with open(f"data/edges/{dataset}.edges", "r") as f:
        for line in f:
            tokens = line.split()
            u = int(tokens[0])
            v = int(tokens[1])
            G.add_edge(u, v)
    return G

def load_poincare_embeddings(dataset, dim):
    """Loads pre-trained hyperbolic embeddings for a given dataset and dimension."""
    embeddings_path = f"data/embeddings/{dataset}_{dim}_poincare.npy"
    return np.load(embeddings_path)
