import torch.nn as nn


def feed_forward_network(d_model, hidden_dim):
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, d_model)
    )
