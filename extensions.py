import torch
from torch.nn import functional as F


@torch.no_grad()
def hebbian_update(projection_matrix, features, lr=0.1, epochs=1, hebb_k=0.01, delta=0.0, batch_size=256):
    """ Task 1: Activity-Dependent Learnable Sparse Projection.

    The non-zero pattern of the Fly-CL projection matrix is kept as a fixed binary connectivity
    mask; only the weights on these connections are adapted with a local Hebbian / anti-Hebbian
    rule in the spirit of BioHash (Ryali et al., 2020):

        dW_mu = g_mu * (mask_mu * x - <W_mu, x> W_mu)

    where g_mu = 1 for the hebb_k most active units of a sample (Hebbian), g_mu = -delta for the
    next hebb_k units (anti-Hebbian) and 0 otherwise. Each row keeps the norm it had at
    initialization, so the expanded features stay on the scale of the original Fly-CL.

    projection_matrix: sparse (expand_dim, embedding_dim), features: (N, embedding_dim)
    """
    W = projection_matrix.to_dense()
    mask = (W != 0).to(W.dtype)
    row_norm = W.norm(dim=1, keepdim=True)
    W = W / row_norm

    X = F.normalize(features - features.mean(dim=0, keepdim=True), dim=1)
    k = max(1, int(W.size(0) * hebb_k))
    for _ in range(epochs):
        for start in range(0, X.size(0), batch_size):
            x = X[start:start + batch_size]
            activity = x @ W.T  # B, expand_dim
            indices = activity.topk(2 * k if delta > 0 else k, dim=1)[1]
            g = torch.zeros_like(activity)
            g.scatter_(1, indices[:, :k], 1.0)
            if delta > 0:
                g.scatter_(1, indices[:, k:], -delta)
            dW = (g.T @ x) * mask - (g * activity).sum(dim=0)[:, None] * W
            count = g.abs().sum(dim=0).clamp(min=1.0)[:, None]  # activity-dependent step size
            W = F.normalize(W + lr * dW / count, dim=1)

    return (W * mask * row_norm).to_sparse_csc()

