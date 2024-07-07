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
) -> float:
    """
    Compute the Centered Kernel Alignment between two given matrices. Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.
    :param x: tensor of shape (n, j).
    :param y: tensor of shape (n, k).
    :param kernel: the kernel used to compute the Gram matrices, must be "linear" or "rbf" (default="linear).
    :param unbiased: whether to use the unbiased version of CKA (default=False).
    :param threshold: the threshold used by the RBF kernel (default=1.0).
    :param method: the method used to compute the CKA value, must be "fro_norm" (Frobenius norm) or "hsic"
        (Hilbert-Schmidt Independence Criterion). Note that the choice does not influence the output
        (default="fro_norm").
    :return: a float in [0, 1] that is the CKA value between the two given matrices.
    """
    assert kernel in ["linear", "rbf"]
    assert method in ["hsic", "fro_norm"]

    x = x.type(torch.float64) if not x.dtype == torch.float64 else x
    y = y.type(torch.float64) if not y.dtype == torch.float64 else y

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
    """
    Compute the minibatch version of CKA from Nguyen et al. (https://arxiv.org/abs/2010.15327). This computation is
    performed with linear kernel and by calculating HSIC_1, it is based on the one implemented by
    https://github.com/numpee/CKA.pytorch/blob/07874ec7e219ad29a29ee8d5ebdada0e1156cf9f/cka.py#L107.
    :param x: tensor of shape (bsz, n, j).
    :param y: tensor of shape (bsz, n, k).
    :return: a float in [0, 1] that is the CKA value between the two given tensors.
    """
    x = x.type(torch.float64) if not x.dtype == torch.float64 else x
    y = y.type(torch.float64) if not y.dtype == torch.float64 else y

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