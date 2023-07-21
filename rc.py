import torch.nn as nn


class RC(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(RC, self).__init__()
        self.LN = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, output):
        return self.LN(query + self.dropout(output))
