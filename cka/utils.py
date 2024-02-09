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


def center_matrix(x: torch.Tensor, unbiased=False) -> torch.Tensor:
    """
    Centers a given Gram matrix. Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.
    :param x: tensor of shape (n, n).
    :param unbiased: whether to use the unbiased version of the centering (default=False).
    :return: the centered version of the given Gram matrix.
    """
    if not torch.allclose(x, x.T):
        raise ValueError("The given matrix must be symmetric.")

    x = x.detach().clone()
    if unbiased:
        n = x.shape[0]
        x.fill_diagonal_(0)
        means = torch.sum(x, dim=0, dtype=torch.float64) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        x -= means[:, None]
        x -= means[None, :]
        x.fill_diagonal_(0)
    else:
        means = torch.mean(x, dim=0, dtype=torch.float64)
        means -= torch.mean(means) / 2
        x -= means[:, None]
        x -= means[None, :]

    return x


def hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hilbert-Schmidt Independence Criterion on two given Gram matrices.
    :param x: matrix of shape (n, n)
    :param y: matrix of shape (n, n)
    :return: the Hilbert-Schmidt Independence Criterion values.
    """
    if not torch.allclose(x, x.T) and not torch.allclose(y, y.T):
        raise ValueError("The given matrices must be symmetric.")

    # Build the identity matrix
    n = x.shape[0]
    identity = torch.eye(n, n, dtype=x.dtype, device=x.device)

    # Build the centering matrix
    h = identity - torch.ones(n, n, dtype=x.dtype, device=x.device) / n

    # Compute x * h and y * h
    xh = torch.mm(x, h)
    yh = torch.mm(y, h)

    # Compute the trace of the product xh * lh
    trace = torch.trace(xh.mm(yh))
    return trace / (n - 1) ** 2


def cka_base(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: str = "linear",
    threshold: float = 1.0,
    method: str = "fro_norm",
) -> float:
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
        gram_x = center_matrix(gram_x, False)
        gram_y = center_matrix(gram_y, False)
        norm_xy = gram_x.ravel().dot(gram_y.ravel())
        norm_xx = torch.linalg.norm(gram_x, ord="fro")
        norm_yy = torch.linalg.norm(gram_y, ord="fro")
        cka = norm_xy / (norm_xx * norm_yy)

    return cka
