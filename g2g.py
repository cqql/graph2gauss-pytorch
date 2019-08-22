import random

import networkx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def load_dataset(filename):
    def load_sparse_matrix(file, name):
        return sp.csr_matrix(
            (file[f"{name}_data"], file[f"{name}_indices"], file[f"{name}_indptr"]),
            shape=file[f"{name}_shape"],
        )

    with np.load(filename) as f:
        A = load_sparse_matrix(f, "adj")
        X = load_sparse_matrix(f, "attr")
        z = f["labels"]

        return A, X, z


class Encoder(nn.Module):
    def __init__(self, D, L):
        """Construct the encoder

        Parameters
        ----------
        D : int
            Dimensionality of the node attributes
        L : int
            Dimensionality of the embedding

        """
        super().__init__()

        def xavier_init(layer):
            nn.init.xavier_normal_(layer.weight)
            # TODO: Initialize bias with xavier but pytorch cannot compute the
            # necessary fan-in for 1-dimensional parameters

        self.linear1 = nn.Linear(D, 512)
        self.linear_mu = nn.Linear(512, L)
        self.linear_sigma = nn.Linear(512, L)

        xavier_init(self.linear1)
        xavier_init(self.linear_mu)
        xavier_init(self.linear_sigma)

    def forward(self, node):
        h = F.relu(self.linear1(node))
        mu = self.linear_mu(h)
        sigma = F.elu(self.linear_sigma(h)) + 1

        return mu, sigma


def level_sets(A):
    G = networkx.from_scipy_sparse_matrix(A)
    paths = networkx.all_pairs_shortest_path(G)

    def reduce_paths(paths):
        max_depth = max(len(path) for path in paths.values())
        levels = [[] for _ in range(max_depth)]

        for node, path in paths.items():
            levels[len(path) - 1].append(node)

        return levels

    return {root: reduce_paths(paths) for root, paths in paths}


def level_counts(A):
    """Count the nodes in each level set for each node

    Parameters
    ----------
    A : np.array
        Adjacency matrix

    Returns
    -------
    { node: [i -> count of level set i] }
    """

    G = networkx.from_scipy_sparse_matrix(A)
    paths = networkx.all_pairs_shortest_path(G)

    def reduce_paths(paths):
        max_depth = max(len(path) for path in paths.values())
        counts = np.zeros(max_depth, dtype=np.int32)

        for path in paths.values():
            counts[len(path) - 1] += 1

        return counts

    return {root: reduce_paths(paths) for root, paths in paths}


def train_test_split(n, train_ratio=0.5):
    nodes = list(range(n))
    split_index = int(n * train_ratio)

    random.shuffle(nodes)
    return nodes[:split_index], nodes[split_index:]


def sample_edges(level_sets, level_counts, size=1):
    if len(level_sets) < 2:
        raise Exception("1-partite graphs contain no edges")

    n = level_counts.sum()

    node_map = [node for level_set in level_sets for node in level_set]

    # Store the cardinality of the level set of each node
    n_i = np.array(
        [
            count
            for level_set, count in zip(level_sets, level_counts)
            for node in level_set
        ]
    )

    # Precompute the start of each level set in the node index set 1..n
    start_i = np.array(
        [
            end - count
            for level_set, count, end in zip(
                level_sets, level_counts, level_counts.cumsum()
            )
            for node in level_set
        ]
    )

    # Each node has edges to every other node except the ones in its own level
    # set
    out_degrees = np.full(n, n) - n_i

    # Sample nodes proportionally to their out-degree
    p = out_degrees / out_degrees.sum()
    source = np.random.choice(n, size=size, p=p, replace=True)

    # For every source node, select one outgoing edge uniformly at random
    dest = np.empty_like(source)
    for i in range(size):
        s = source[i]

        # Sample from 1..(start_i[s]..+n_i[s]-1)..n
        d = np.random.randint(out_degrees[s])
        if d >= start_i[s]:
            d += n_i[s]

        dest[i] = d

    edges = np.stack([source, dest], axis=1)

    # Translate node indices back into entries from level_sets
    for i in range(size):
        edges[i][0] = node_map[edges[i][0]]
        edges[i][1] = node_map[edges[i][1]]

    return edges


def main():
    np.random.seed(0)

    A, X, z = load_dataset("citeseer.npz")
    n = A.shape[0]

    train_nodes, test_nodes = train_test_split(n)

    A_train = A[train_nodes, :][:, train_nodes].astype(np.float32)
    X_train = X[train_nodes].astype(np.float32)
    A_test = A[test_nodes, :][:, test_nodes]
    X_test = X[test_nodes]

    L = level_sets(A_train)
    N = {node: np.array(list(map(len, ls))) for node, ls in L.items()}
    # Precompute the multiplier for each node in the loss function
    F = np.array([N[i][1:].sum() ** 2 - (N[i][1:] ** 2).sum() for i in range(len(N))])

    L_dim = 10
    encoder = Encoder(X.shape[1], L_dim)
    optimizer = optim.Adam(encoder.parameters())

    nsamples = 3
    epochs = 5
    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = torch.tensor(0.0)
        for i in range(A_train.shape[0]):
            # If a node only has first-degree neighbors, the loss is undefined
            if len(N[i]) < 3:
                continue

            E = torch.tensor(0.0)
            edges = sample_edges(L[i][1:], N[i][1:], size=nsamples)
            for edge in edges:
                j, k = edge[0], edge[1]

                mu, sigma = encoder(torch.tensor(X_train[[i, j, k]].toarray()))
                mu_i, sigma_i = mu[0], sigma[0]
                mu_j, sigma_j = mu[1], sigma[1]
                mu_k, sigma_k = mu[2], sigma[2]

                diff_ij = mu_i - mu_j
                closer = (
                    0.5 * (sigma_j / sigma_i).sum()
                    + (diff_ij / sigma_i).dot(diff_ij)
                    - L_dim
                    - torch.log((sigma_j / sigma_i)).sum()
                )
                diff_ik = mu_i - mu_k
                apart = (
                    -0.5 * (sigma_k / sigma_i).sum()
                    + (diff_ik / sigma_i).dot(diff_ik)
                    - L_dim
                )

                E += closer ** 2 + torch.exp(apart) * (
                    (sigma_k / sigma_i).prod() + np.exp(0.5)
                )
            E /= nsamples
            loss += F[i] * E
        loss *= 0.5

        loss.backward()

        print(f"Epoch {epoch:2d} - Loss {loss:.3f}")

        optimizer.step()


if __name__ == "__main__":
    main()
