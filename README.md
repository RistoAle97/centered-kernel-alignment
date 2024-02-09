<div align="center">

# CKA Pytorch
**CKA (Centered Kernel Alignment) in Pytorch.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch)

---
</div>

## About
CKA [1] is a similarity metric between representations of features in neural networks, based on the Hilbert-Schmidt Independence Criterion (HSIC) [2].  
Given two matrices $X \in \mathbb{R}^{n\times s_1}$ and $Y \in \mathbb{R}^{n\times s_2}$ representing the output of two layers, we can define two auxiliary $n \times n$ Gram matrices $K=XX^T$ and $L=YY^T$ and compute the *dot-product similarity* between them
$$\langle vec(XX^T), vec(YY^T)\rangle = tr(XX^T YY^T) = \lVert Y^T X \rVert_F^2.$$
Then, the HSIC on $K$ and $L$ is defined as
$$HSIC(K, L) = \frac{tr(KHLH)}{(n - 1)^2},$$
where $H = I_n - \frac{1}{n}J_n$ is the centering matrix and $J_n$ is an $n \times n$ matrix filled with ones. Finally, to obtain the CKA value we only need to normalize HSIC
$$CKA(K, L) = \frac{HSIC(K, L)}{\sqrt{HSIC(K, K) HSIC(L, L)}}.$$

## Installation
This project requires python >= 3.9. All the necessary packages can be installed with
```bash
pip install -r requirements.txt
```
Take a look at `main.py` for a simple use case.

---

## References
[1] Kornblith, Simon, et al. "Similarity of neural network representations revisited." *International Conference on Machine Learning*. PMLR, 2019.

[2] Wang, Tinghua et al. "Learning with hilbert-schmidt independence criterion: A review and new perspectives." *Knowl. Based Syst*., 2021. 

## License
This project is MIT licensed.
