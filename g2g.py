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

    def nodes(self):
        return range(self.A.shape[0])

    def eligible_nodes(self):
        """Nodes that can be used to compute the loss"""
        N = self.level_counts

        # If a node only has first-degree neighbors, the loss is undefined
        return [i for i in self.nodes() if len(N[i]) >= 3]

    def sample_two_neighbors(self, node, size=1):
        level_sets = self.level_sets[node]
        if len(level_sets) < 2:
            raise Exception("1-partite graphs contain no edges")

        level_counts = self.level_counts[node]
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

        # Translate node indices back into entries from level_sets
        for i in range(size):
            source[i] = node_map[source[i]]
            dest[i] = node_map[dest[i]]

        return source, dest



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
        weights = graph.loss_weights

        mu, sigma = self.forward(torch.tensor(X.toarray()))
        loss = torch.tensor(0.0)
        for i in graph.eligible_nodes():
            mu_i, sigma_i = (
                mu[i].expand((nsamples, -1)),
                sigma[i].expand((nsamples, -1)),
            )

            # MC-estimate the loss function of this nodes' neighborhood graph
            js, ks = graph.sample_two_neighbors(i, size=nsamples)

            mu_j, sigma_j = mu[js], sigma[js]
            mu_k, sigma_k = mu[ks], sigma[ks]

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

            E = (
                closer ** 2 + torch.exp(apart) * torch.sqrt(ratio_ki.prod(axis=-1))
            ).sum() / nsamples

            # Accumulate the weighted sum of expectations
            loss += weights[i] * E

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

        lr = sklm.LogisticRegressionCV(multi_class="auto", solver="lbfgs", cv=3)
        lr.fit(X_train, z_train)

        f1 = skm.SCORERS["f1_micro"]
        print(f"LR F1 score {f1(lr, X_test, z_test)}")

    trainer.run([train_data], epochs)


if __name__ == "__main__":
    main()
