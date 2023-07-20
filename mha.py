import torch.nn as nn
import math

NEG_INF = -1e9


class MHA(nn.Module):
    def __int__(self, hidden_dim, head_num, dropout_rate=0.1):
        super().__init__()
        assert hidden_dim % head_num == 0, \
            "hidden_dim %s is not divisible by head_num %s" % (hidden_dim, head_num)
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.w_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None, use_dropout=True):
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        bs, q_sl = q.size(0), q.size(1)
        k_sl = v_sl = k.size(1)
        mha_q = q.view(bs, q_sl, self.head_num, -1).transpose(1, 2)
        mha_k = k.view(bs, k_sl, self.head_num, -1).transpose(1, 2)
        mha_v = v.view(bs, v_sl, self.head_num, -1).transpose(1, 2)
        mha_scores = mha_q.matmul(mha_k.transpose(-1, -2)) / math.sqrt(mha_q.size(-1))
        if mask is not None:
            mha_scores.masked_fill_(mask.unsqueeze(1) == True, NEG_INF)
        mha_attn = mha_scores.softmax(-1)
        if use_dropout:
            mha_attn = self.dropout(mha_attn)
        mha_o = mha_attn.matmul(mha_v)
        o = mha_o.transpose(1, 2).contiguous().view(bs, q_sl, self.hidden_dim)
        return self.w_o(o)
