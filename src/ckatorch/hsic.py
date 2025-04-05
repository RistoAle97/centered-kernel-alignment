"""Module for computing HSIC (Hilbert-Schmidt Independence Criterion), both its standard and mini-batch versions."""

import torch


def hsic0(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """Compute the Hilbert-Schmidt Independence Criterion on two given Gram matrices.

    Args:
        gram_x (torch.Tensor): Gram matrix of shape (n, n), this is equivalent to K from the original paper.
        gram_y (torch.Tensor): Gram matrix of shape (n, n), this is equivalent to L from the original paper.

    Returns:
        torch.Tensor: a tensor with the Hilbert-Schmidt Independence Criterion values.

    Raises:
        ValueError: if ``gram_x`` and ``gram_y`` are not symmetric.
    """
    if not torch.allclose(gram_x, gram_x.T) and not torch.allclose(gram_y, gram_y.T):
        raise ValueError("The given matrices must be symmetric.")

    # Build the identity matrix
    n = gram_x.shape[0]
    identity = torch.eye(n, n, dtype=gram_x.dtype, device=gram_x.device)

    # Build the centering matrix
    h = identity - torch.ones(n, n, dtype=gram_x.dtype, device=gram_x.device) / n

    # Compute k * h and l * h
    kh = torch.mm(gram_x, h)
    lh = torch.mm(gram_y, h)

    # Compute the trace of the product kh * lh
    trace = torch.trace(kh.mm(lh))
    return trace / (n - 1) ** 2


def hsic1(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """Compute the batched version of the Hilbert-Schmidt Independence Criterion on Gram matrices.

    This version is based on
    https://github.com/numpee/CKA.pytorch/blob/07874ec7e219ad29a29ee8d5ebdada0e1156cf9f/cka.py#L107.

    Args:
        gram_x (torch.Tensor): batch of Gram matrices of shape (bsz, n, n).
        gram_y (torch.Tensor): batch of Gram matrices of shape (bsz, n, n).

    Returns:
        torch.Tensor: a tensor with the unbiased Hilbert-Schmidt Independence Criterion values.

    Raises:
        ValueError: if ``gram_x`` and ``gram_y`` do not have the same shape or if they do not have exactly three
            dimensions.
    """
    if len(gram_x.size()) != 3 or gram_x.size() != gram_y.size():
        raise ValueError("Invalid size for one of the two input tensors.")

    n = gram_x.shape[-1]
    gram_x = gram_x.clone()
    gram_y = gram_y.clone()

    # Fill the diagonal of each matrix with 0
    gram_x.diagonal(dim1=-1, dim2=-2).fill_(0)
    gram_y.diagonal(dim1=-1, dim2=-2).fill_(0)

    # Compute the product between k (i.e.: gram_x) and l (i.e.: gram_y)
    kl = torch.bmm(gram_x, gram_y)

    # Compute the trace (sum of the elements on the diagonal) of the previous product, i.e.: the left term
    trace_kl = kl.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)

    # Compute the middle term
    middle_term = gram_x.sum((-1, -2), keepdim=True) * gram_y.sum((-1, -2), keepdim=True)
    middle_term /= (n - 1) * (n - 2)

    # Compute the right term
    right_term = kl.sum((-1, -2), keepdim=True)
    right_term *= 2 / (n - 2)

    # Put all together to compute the main term
    main_term = trace_kl + middle_term - right_term

    # Compute the hsic values
    out = main_term / (n**2 - 3 * n)
    return out.squeeze(-1).squeeze(-1)
