import torch.nn as nn
from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model)
        )

        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        # Self-attention block (Pre-LN)
        x = x + self.dropout(self.attn(self.ln1(x), mask))

        # Feed-forward block (Pre-LN)
        x = x + self.dropout(self.ff(self.ln2(x)))

        return x