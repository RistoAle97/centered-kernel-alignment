<div align="center">

# CKA Pytorch
**CKA (Centered Kernel Alignment) in Pytorch.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch)

---
</div>

## :black_nib: About
> [!NOTE]
> Centered Kernel Alignment (CKA) [1] is a similarity index between representations of features in neural networks, based on the Hilbert-Schmidt Independence Criterion (HSIC) [2]. Given a set of examples, CKA compares the representations of examples passed through the layers that we want to compare.

Given two matrices $X \in \mathbb{R}^{n\times s_1}$ and $Y \in \mathbb{R}^{n\times s_2}$ representing the output of two layers, we can define two auxiliary $n \times n$ Gram matrices $K=XX^T$ and $L=YY^T$ and compute the *dot-product similarity* between them
$$\langle vec(XX^T), vec(YY^T)\rangle = tr(XX^T YY^T) = \lVert Y^T X \rVert_F^2.$$
Then, the HSIC on $K$ and $L$ is defined as
$$HSIC(K, L) = \frac{tr(KHLH)}{(n - 1)^2},$$
where $H = I_n - \frac{1}{n}J_n$ is the centering matrix and $J_n$ is an $n \times n$ matrix filled with ones. Finally, to obtain the CKA value we only need to normalize HSIC
$$CKA(K, L) = \frac{HSIC(K, L)}{\sqrt{HSIC(K, K) HSIC(L, L)}}.$$

---

## :package: Installation
This project requires python >= 3.9. All the necessary packages can be installed with
```bash
pip install -r requirements.txt
```
Take a look at `main.py` for a simple use case.

---

## :framed_picture:	Plots
<div align="center">
  <img src="https://github.com/RistoAle97/centered-kernel-alignment/blob/main/plots/model_comparison_itself.png" alt="Model compared with itself" height=310/>
  <img src="https://github.com/RistoAle97/centered-kernel-alignment/blob/main/plots/model_comparison.png" alt="Model comparison" height=310/>
</div>

---

## :books: Bibliography
[1] Kornblith, Simon, et al. ["Similarity of neural network representations revisited."](https://arxiv.org/abs/1905.00414) *International Conference on Machine Learning*. PMLR, 2019.

[2] Wang, Tinghua, Xiaolu Dai, and Yuze Liu. ["Learning with Hilbert–Schmidt independence criterion: A review and new perspectives."](https://www.sciencedirect.com/science/article/pii/S0950705121008297) *Knowledge-based systems* 234 (2021): 107567.

This project is also based on the following works:
- https://github.com/google-research/google-research/tree/master/representation_similarity (original implementantion).
- https://github.com/AntixK/PyTorch-Model-Compare (nice PyTorch implementation that employs hooks).

---

## :memo: License
This project is [MIT licensed](https://github.com/RistoAle97/centered-kernel-alignment/blob/main/LICENSE).
