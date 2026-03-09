from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    max_seq_len: int
    dropout: float