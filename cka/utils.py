from typing import Literal

import torch


def linear_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gram (kernel) matrix for a linear kernel. Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.
    :param x: tensor of shape (n, m).
    :return: tensor of shape (n, n).
    """
    return torch.mm(x, x.T)


def rbf_kernel(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """
    Computes the Gram (kernel) matrix for an RBF kernel. Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.
    :param x: tensor of shape (n, m).
    :param threshold: fraction of median Euclidean distance to use as RBF kernel bandwidth (default=1.0).
    :return: tensor of shape (n, n).
    """
    dot_products = torch.mm(x, x.T)
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold**2 * sq_median_distance))


def center_gram_matrix(gram_matrix: torch.Tensor, unbiased=False) -> torch.Tensor:
    """
    Centers a given Gram matrix. Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.
    :param gram_matrix: tensor of shape (n, n).
    :param unbiased: whether to use the unbiased version of the centering (default=False).
    :return: the centered version of the given Gram matrix.
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


def hsic(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hilbert-Schmidt Independence Criterion on two given Gram matrices.
    :param gram_x: Gram matrix of shape (n, n), this is equivalent to K from the original paper.
    :param gram_y: Gram matrix of shape (n, n), this is equivalent to L from the original paper.
    :return: the Hilbert-Schmidt Independence Criterion values.
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
        (Hilbert-Schmidt Independence Criterion). Note that the choice does not have influence on the output
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
        hsic_xy = hsic(gram_x, gram_y)
        hsic_xx = hsic(gram_x, gram_x)
        hsic_yy = hsic(gram_y, gram_y)
        cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
    else:
        gram_x = center_gram_matrix(gram_x, unbiased)
        gram_y = center_gram_matrix(gram_y, False)
        norm_xy = gram_x.ravel().dot(gram_y.ravel())
        norm_xx = torch.linalg.norm(gram_x, ord="fro")
        norm_yy = torch.linalg.norm(gram_y, ord="fro")
        cka = norm_xy / (norm_xx * norm_yy)

    return cka


def hsic1(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """
    Compute the batched version of the Hilbert-Schmidt Independence Criterion on Gram matrices.
    :param gram_x: a tensor of shape (bsz, n, n).
    :param gram_y: a tensor of shape (bsz, n, n).
    :return: the unbiased Hilbert-Schmidt Independence Criterion values.
    """
    assert len(gram_x.size()) == 3 and gram_x.size() == gram_y.size()
    n = gram_x.shape[-1]
    gram_x = gram_x.clone()
    gram_y = gram_y.clone()

    # Fill the diagonal of each matrix with 0
    gram_x.diagonal(dim1=-1, dim2=-2).fill_(0)
    gram_y.diagonal(dim1=-1, dim2=-2).fill_(0)

    # Compute the product between k and l
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
    out = main_term / (n ** 2 - 3 * n)
    return out.squeeze(-1).squeeze(-1)


def batched_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
