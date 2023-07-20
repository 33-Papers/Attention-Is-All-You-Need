import torch.nn as nn


class FFN(nn.Module):
    def __int__(self, hidden_dim, ffn_dim, dropout_rate=0.1, fc_or_conv='fc'):
        super().__int__()
        self.fc_or_conv = fc_or_conv
        if self.fc_or_conv == 'fc':
            self.k = nn.Linear(hidden_dim, ffn_dim)
            self.v = nn.Linear(ffn_dim, hidden_dim)
        elif self.fc_or_conv == 'conv':
            self.k = nn.Conv1d(in_channels=hidden_dim, out_channels=ffn_dim, kernel_size=1)
            self.v = nn.Conv1d(in_channels=hidden_dim, out_channels=ffn_dim, kernel_size=1)
        else:
            raise ValueError('fc_or_conv must be in ["fc", "conv"] but %s' % fc_or_conv)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q):
        if self.fc_or_conv == 'fc':
            attn = self.k(q)
            attn = self.dropout(self.act(attn))
            o = self.v(attn)
        elif self.fc_or_conv == 'conv':
            attn = self.k(q.transpose(-2, -1))
            attn = self.dropout(self.act(attn))
            o = self.v(attn).transpose(-2, -1)
        else:
            raise ValueError('fc_or_conv must in ["fc", "conv"] but %s' % self.fc_or_conv)

        return o
