"""Module that implements both base and mini-batch CKA."""

from typing import Literal

import torch

from .hsic import hsic0, hsic1
from .utils import center_gram_matrix, linear_kernel, rbf_kernel


def cka_base(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: Literal["linear", "rbf"] = "linear",
    unbiased: bool = False,
    threshold: float = 1.0,
    method: Literal["fro_norm", "hsic"] = "fro_norm",
) -> torch.Tensor:
    """Computes the Centered Kernel Alignment (CKA) between two given matrices.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        x (torch.Tensor): tensor of shape (n, j).
        y (torch.Tensor): tensor of shape (n, k).
        kernel (Literal["linear", "rbf"]): the kernel used to compute the Gram matrices, must be "linear" or "rbf"
            (default="linear").
        unbiased (bool): whether to use the unbiased version of CKA (default=False).
        threshold (float): the threshold used by the RBF kernel (default=1.0).
        method (Literal["fro_norm", "hsic"]): the method used to compute the CKA value, must be "fro_norm"
            (Frobenius norm) or "hsic" (Hilbert-Schmidt Independence Criterion). Note that the choice does not
            influence the output (default="fro_norm").

    Returns:
        torch.Tensor: a float tensor in [0, 1] that is the CKA value between the two given matrices.

    Raises:
        ValueError: if ``kernel`` is not "linear" or "rbf" or if ``method`` is not "fro_norm" or "hsic".
    """
    if kernel not in ["linear", "rbf"]:
        raise ValueError("The chosen kernel must be either 'linear' or 'rbf'.")

    if method not in ["hsic", "fro_norm"]:
        raise ValueError("The chosen method must be either 'hsic' or 'fro_norm'.")

    x = x.type(torch.float64) if x.dtype != torch.float64 else x
    y = y.type(torch.float64) if y.dtype != torch.float64 else y

    # Build the Gram matrices by applying the kernel
    gram_x = linear_kernel(x) if kernel == "linear" else rbf_kernel(x, threshold)
    gram_y = linear_kernel(y) if kernel == "linear" else rbf_kernel(y, threshold)

    # Compute CKA by either using HSIC or the Frobenius norm
    if method == "hsic":
        hsic_xy = hsic0(gram_x, gram_y)
        hsic_xx = hsic0(gram_x, gram_x)
        hsic_yy = hsic0(gram_y, gram_y)
        cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
    else:
        gram_x = center_gram_matrix(gram_x, unbiased)
        gram_y = center_gram_matrix(gram_y, unbiased)
        norm_xy = gram_x.ravel().dot(gram_y.ravel())
        norm_xx = torch.linalg.norm(gram_x, ord="fro")
        norm_yy = torch.linalg.norm(gram_y, ord="fro")
        cka = norm_xy / (norm_xx * norm_yy)

    return cka


def cka_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the minibatch version of CKA from Nguyen et al. (https://arxiv.org/abs/2010.15327).

    This computation is performed with linear kernel and by calculating HSIC_1.

    Args:
        x (torch.Tensor): tensor of shape (bsz, n, j).
        y (torch.Tensor): tensor of shape (bsz, n, k).

    Returns:
        torch.Tensor: a float tensor in [0, 1] that is the CKA value between the two given tensors.
    """
    x = x.type(torch.float64) if x.dtype != torch.float64 else x
    y = y.type(torch.float64) if y.dtype != torch.float64 else y

    # Build the Gram matrices by applying the linear kernel
    gram_x = torch.bmm(x, x.transpose(1, 2))
    gram_y = torch.bmm(y, y.transpose(1, 2))

    # Compute the HSIC values for the entire batches
    hsic1_xy = hsic1(gram_x, gram_y)
    hsic1_xx = hsic1(gram_x, gram_x)
    hsic1_yy = hsic1(gram_y, gram_y)

    # Compute the CKA value
    cka = hsic1_xy.sum() / (hsic1_xx.sum() * hsic1_yy.sum()).sqrt()
    return cka
