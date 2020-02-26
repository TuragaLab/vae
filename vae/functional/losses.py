"""This is a functional counterpart of the vae.losses module.

Implemented functional losses:
    vimco, p_wake, q_wake, q_sleep
"""

import torch

def vimco(log_q, log_p):
    """Functional VIMCO loss.

    VIMCO paper: https://arxiv.org/abs/1602.06725

    Args:
        log_q (tensor): log q(z|x), (batch_size, n_samples).
        log_p (tensor): log p(x,z), (batch_size, n_samples).

    Returns:
        scalar VIMCO loss.
    """
    batch_size, n_samples = log_q.shape
    K = torch.tensor(n_samples)
    log_f = log_p - log_q
    log_fh = (torch.sum(log_f, dim=1).unsqueeze(1)-log_f) / (K-1)
    Log_f = (torch.ones(batch_size, K, K) - torch.eye(K).unsqueeze(0)) * log_f.unsqueeze(1) + torch.diag_embed(log_fh)
    L = torch.logsumexp(log_f, dim=1) - torch.log(K.float())
    Li = torch.logsumexp(Log_f, dim=2) - torch.log(K.float())
    w = torch.nn.functional.softmax(log_f, dim=1)
    objective = w.detach()*log_f + (L.unsqueeze(1)-Li).detach()*log_q # (batch_size, n_samples)
    loss = -torch.mean(objective) # scalar loss
    return loss

def p_wake(log_q, log_p):
    pass

def q_wake(log_q, log_p):
    pass

def q_sleep(log_q): # Trivial, but for completeness...
    pass
