import torch


def get_mask_from_lengths(seq_lengths, max_seq_len, device):
    """convert an array of minibatch sequence lengths into 2D boolean masks"""
    mask = torch.arange(max_seq_len).expand(len(seq_lengths), max_seq_len) >= seq_lengths.unsqueeze(1)
    mask.to(device)
    
    return mask