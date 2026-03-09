import torch
from model.transformer import TransformerModel


def load_checkpoint(path, config, device='cpu'):
    model = TransformerModel(config).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()
    return model