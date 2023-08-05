import torch
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    key_dim = torch.tensor(key.size(-1), dtype=torch.float32)
    scaled_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(key_dim)

    if mask is not None:
        scaled_scores = torch.where(mask == 0, float('-inf'), scaled_scores)

    weights = F.softmax(scaled_scores, dim=-1)
    output = torch.matmul(weights, value)
    return output, weights
