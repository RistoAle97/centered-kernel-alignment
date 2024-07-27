"""Utilities for computing and centering Gram matrices."""

import torch


def linear_kernel(x: torch.Tensor) -> torch.Tensor:
    """Computes the Gram (kernel) matrix for a linear kernel.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        x: tensor of shape (n, m).

    Returns:
        a Gram matrix which is a tensor of shape (n, n).
    """
    return torch.mm(x, x.T)


def rbf_kernel(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Computes the Gram (kernel) matrix for an RBF kernel.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        x: tensor of shape (n, m).
        threshold: fraction of median Euclidean distance to use as RBF kernel bandwidth (default=1.0).

    Returns:
        a Gram matrix which is a tensor of shape (n, n).
    """
    dot_products = torch.mm(x, x.T)
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold**2 * sq_median_distance))


def center_gram_matrix(gram_matrix: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    """Centers a given Gram matrix.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        gram_matrix: tensor of shape (n, n).
        unbiased: whether to use the unbiased version of the centering (default=False).

    Returns:
        the centered version of the given Gram matrix.
    """
    if not torch.allclose(gram_matrix, gram_matrix.T):
        raise ValueError("The given matrix must be symmetric.")

    gram_matrix = gram_matrix.detach().clone()
    if unbiased:
        n = gram_matrix.shape[0]
        gram_matrix.fill_diagonal_(0)
        means = torch.sum(gram_matrix, dim=0, dtype=torch.float64) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram_matrix -= means[:, None]
        gram_matrix -= means[None, :]
        gram_matrix.fill_diagonal_(0)
    else:
        means = torch.mean(gram_matrix, dim=0, dtype=torch.float64)
        means -= torch.mean(means) / 2
        gram_matrix -= means[:, None]
        gram_matrix -= means[None, :]

    return gram_matrix
