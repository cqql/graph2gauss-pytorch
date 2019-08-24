import argparse
import random

import networkx
import numpy as np
import scipy.sparse as sp
import sklearn.linear_model as sklm
import sklearn.metrics as skm
import sklearn.model_selection as skms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader

CHECKPOINT_PREFIX = "g2g"


def load_dataset(filename):
    def load_sparse_matrix(file, name):
        return sp.csr_matrix(
            (file[f"{name}_data"], file[f"{name}_indices"], file[f"{name}_indptr"]),
            shape=file[f"{name}_shape"],
            dtype=np.float32,
        )

    with np.load(filename) as f:
        A = load_sparse_matrix(f, "adj")
        X = load_sparse_matrix(f, "attr")
        z = f["labels"].astype(np.float32)

        return A, X, z


class CompleteKPartiteGraph:
    """A complete k-partite graph
    """

    def __init__(self, partitions):
        """
        Parameters
        ----------
        partitions : [[int]]
            List of node partitions where each partition is list of node IDs
        """

        self.partitions = partitions
        self.counts = np.array([len(p) for p in partitions])
        self.total = self.counts.sum()

        assert len(self.partitions) >= 2
        assert np.all(self.counts > 0)

        # Enumerate all nodes so that we can easily look them up with an index
        # from 1..total
        self.nodes = np.array([node for partition in partitions for node in partition])

        # Precompute the partition count of each node
        self.n_i = np.array(
            [n for partition, n in zip(self.partitions, self.counts) for _ in partition]
        )

        # Precompute the start of each node's partition in self.nodes
        self.start_i = np.array(
            [
                end - n
                for partition, n, end in zip(
                    self.partitions, self.counts, self.counts.cumsum()
                )
                for node in partition
            ]
        )

        # Each node has edges to every other node except the ones in its own
        # level set
        self.out_degrees = np.full(self.total, self.total) - self.n_i

        # Sample the first nodes proportionally to their out-degree
        self.p = self.out_degrees / self.out_degrees.sum()

    def sample_edges(self, size=1):
        """Sample edges (j, k) from this graph uniformly and independently

        Returns
        -------
        ([j], [k])
        j will always be in a lower partition than k
        """

        # Sample the originating nodes for each edge
        j = np.random.choice(self.total, size=size, p=self.p, replace=True)

        # For each j sample one outgoing edge uniformly
        #
        # Se we want to sample from 1..n \ start[j]...(start[j] + count[j]). We
        # do this by sampling from 1..#degrees[j] and if we hit a node

        k = np.random.randint(self.out_degrees[j])
        filter = k >= self.start_i[j]
        k += filter.astype(np.int) * self.n_i[j]

        # Swap nodes such that the partition index of j is less than that of k
        # for each edge
        wrong_order = k < j
        tmp = k[wrong_order]
        k[wrong_order] = j[wrong_order]
        j[wrong_order] = tmp

        # Translate node indices back into user configured node IDs
        j = self.nodes[j]
        k = self.nodes[k]

        return j, k


class AttributedGraph:
    def __init__(self, A, X, z):
        self.A = A
        self.X = X
        self.z = z
        self.level_sets = level_sets(A)

        # Precompute the cardinality of each level set for every node
        self.level_counts = {
            node: np.array(list(map(len, level_sets)))
            for node, level_sets in self.level_sets.items()
        }

        # Precompute the weights of each node's expected value in the loss
        N = self.level_counts
        self.loss_weights = 0.5 * np.array(
            [N[i][1:].sum() ** 2 - (N[i][1:] ** 2).sum() for i in self.nodes()]
        )

        n = self.A.shape[0]
        self.neighborhoods = [None] * n
        for i in range(n):
            ls = self.level_sets[i]
            if len(ls) >= 3:
                self.neighborhoods[i] = CompleteKPartiteGraph(ls[1:])

    def nodes(self):
        return range(self.A.shape[0])

    def eligible_nodes(self):
        """Nodes that can be used to compute the loss"""
        N = self.level_counts

        # If a node only has first-degree neighbors, the loss is undefined
        return [i for i in self.nodes() if len(N[i]) >= 3]

    def sample_two_neighbors(self, node, size=1):
        """Sample to nodes from the neighborhood of different rank"""

        level_sets = self.level_sets[node]
        if len(level_sets) < 3:
            raise Exception(f"Node {node} has only one layer of neighbors")

        return self.neighborhoods[node].sample_edges(size)


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

        self.D = D
        self.L = L

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

    def compute_loss(self, graph, nsamples):
        """Compute the energy-based loss from the paper
        """

        X = graph.X

        mu, sigma = self.forward(torch.tensor(X.toarray()))

        eligible_nodes = list(graph.eligible_nodes())
        nrows = len(eligible_nodes) * nsamples
        mu_i = torch.empty((nrows, mu.shape[-1]))
        sigma_i = torch.empty((nrows, sigma.shape[-1]))
        mu_j = torch.empty((nrows, mu.shape[-1]))
        sigma_j = torch.empty((nrows, sigma.shape[-1]))
        mu_k = torch.empty((nrows, mu.shape[-1]))
        sigma_k = torch.empty((nrows, sigma.shape[-1]))

        weights = torch.empty(nrows)

        for index, i in enumerate(eligible_nodes):
            start = index * nsamples
            end = start + nsamples
            mu_i[start:end], sigma_i[start:end] = mu[i], sigma[i]

            js, ks = graph.sample_two_neighbors(i, size=nsamples)
            mu_j[start:end], sigma_j[start:end] = mu[js], sigma[js]
            mu_k[start:end], sigma_k[start:end] = mu[ks], sigma[ks]

            weights[start:end] = graph.loss_weights[i]

        diff_ij = mu_i - mu_j
        ratio_ji = sigma_j / sigma_i
        closer = 0.5 * (
            ratio_ji.sum(axis=-1)
            + ((diff_ij / sigma_i) * diff_ij).sum(axis=-1)
            - self.L
            - torch.log(ratio_ji).sum(axis=-1)
        )

        diff_ik = mu_i - mu_k
        ratio_ki = sigma_k / sigma_i
        apart = -0.5 * (
            ratio_ki.sum(axis=-1)
            + ((diff_ik / sigma_i) * diff_ik).sum(axis=-1)
            - self.L
        )

        E = closer ** 2 + torch.exp(apart) * torch.sqrt(ratio_ki.prod(axis=-1))

        loss = E.dot(weights) / nsamples

        return loss


def level_sets(A):
    """Enumerate the level sets for each node's neighborhood

    Parameters
    ----------
    A : np.array
        Adjacency matrix

    Returns
    -------
    { node: [i -> i-hop neighborhood] }
    """

    G = networkx.from_scipy_sparse_matrix(A)
    paths = networkx.all_pairs_shortest_path(G)

    def reduce_paths(paths):
        max_depth = max(len(path) for path in paths.values())
        levels = [[] for _ in range(max_depth)]

        for node, path in paths.items():
            levels[len(path) - 1].append(node)

        return levels

    return {root: reduce_paths(paths) for root, paths in paths}


def train_test_split(n, train_ratio=0.5):
    nodes = list(range(n))
    split_index = int(n * train_ratio)

    random.shuffle(nodes)
    return nodes[:split_index], nodes[split_index:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("-c", "--checkpoint")
    parser.add_argument("--checkpoints")
    parser.add_argument("dataset")
    args = parser.parse_args()

    epochs = args.epochs
    nsamples = args.samples
    learning_rate = args.lr
    seed = args.seed
    checkpoint_path = args.checkpoint
    checkpoints_path = args.checkpoints
    dataset_path = args.dataset

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    A, X, z = load_dataset(dataset_path)

    n = A.shape[0]
    train_nodes, val_nodes = train_test_split(n, train_ratio=0.5)
    A_train = A[train_nodes, :][:, train_nodes]
    X_train = X[train_nodes]
    z_train = z[train_nodes]
    A_val = A[val_nodes, :][:, val_nodes]
    X_val = X[val_nodes]
    z_val = z[val_nodes]

    train_data = AttributedGraph(A_train, X_train, z_train)
    val_data = AttributedGraph(A_val, X_val, z_val)

    L = 10
    encoder = Encoder(X.shape[1], L)
    if checkpoint_path:
        encoder.load_state_dict(torch.load(checkpoint_path))

    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    def step(engine, dataset):
        optimizer.zero_grad()

        loss = encoder.compute_loss(dataset, nsamples)
        loss.backward()

        optimizer.step()

        return loss.item()

    trainer = Engine(step)

    if checkpoints_path:
        handler = ModelCheckpoint(
            checkpoints_path, CHECKPOINT_PREFIX, n_saved=3, save_interval=1
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"encoder": encoder})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_loss(engine):
        print(f"Epoch {engine.state.epoch:2d} - Loss {engine.state.output:.3f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        if engine.state.epoch % 5 != 1:
            return

        loss = encoder.compute_loss(val_data, nsamples)
        print(f"Validation loss {loss:.3f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def node_classification(engine):
        if engine.state.epoch % 5 != 1:
            return

        X = val_data.X
        z = val_data.z
        X_train, X_test, z_train, z_test = skms.train_test_split(
            X, z, train_size=0.1, stratify=z
        )

        lr = sklm.LogisticRegressionCV(
            multi_class="auto", solver="lbfgs", cv=3, max_iter=500
        )
        lr.fit(X_train, z_train)

        f1 = skm.SCORERS["f1_micro"]
        print(f"LR F1 score {f1(lr, X_test, z_test)}")

    trainer.run([train_data], epochs)


if __name__ == "__main__":
    main()
