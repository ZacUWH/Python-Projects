import json
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or {}

    def encode(self, text):
        tokens = list(text)
        for a, b in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == a and tokens[i+1] == b:
                    tokens[i:i+2] = [a + b]
                else:
                    i += 1
        return [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]

    def decode(self, ids):
        return ''.join(self.id_to_token[i] for i in ids)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'merges': self.merges,
                'special_tokens': self.special_tokens
            }, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(data['vocab'], data['merges'], data.get('special_tokens'))