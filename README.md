<div align="center">

# 🤖 CKA PyTorch 🤖
**CKA (Centered Kernel Alignment) in PyTorch.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://github.com/python/cpython)
[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch)

[![PyPI](https://img.shields.io/pypi/v/ckatorch.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/ckatorch/)
[![Python versions](https://img.shields.io/pypi/pyversions/ckatorch.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/ckatorch/)


</div>

> [!WARNING]
> This repository has been built mainly for personal and academic use since <img height="15" width="15" src="https://cdn.simpleicons.org/pytorch"/>[Captum](https://github.com/pytorch/captum) still needs to implement its variant of CKA. As such, do not expect this project to work for every model.

---

## ✒️ About
> [!NOTE]
> Centered Kernel Alignment (CKA) [1] is a similarity index between representations of features in neural networks, based on the Hilbert-Schmidt Independence Criterion (HSIC) [2]. Given a set of examples, CKA compares the representations of examples passed through the layers that we want to compare.

Given two matrices $X \in \mathbb{R}^{n\times s_1}$ and $Y \in \mathbb{R}^{n\times s_2}$ representing the output of two layers, we can define two auxiliary $n \times n$ Gram matrices $K=XX^T$ and $L=YY^T$ and compute the *dot-product similarity* between them

$$\langle vec(XX^T), vec(YY^T)\rangle = tr(XX^T YY^T) = \lVert Y^T X \rVert_F^2.$$

Then, the $HSIC$ on $K$ and $L$ is defined as

$$HSIC_0(K, L) = \frac{tr(KHLH)}{(n - 1)^2},$$

where $H = I_n - \frac{1}{n}J_n$ is the centering matrix and $J_n$ is an $n \times n$ matrix filled with ones. Finally, to obtain the CKA value we only need to normalize $HSIC_0$

$$CKA(K, L) = \frac{HSIC(K, L)}{\sqrt{HSIC(K, K) HSIC(L, L)}}.$$

> [!NOTE]
> However, naive computation of linear CKA (i.e.: the previous equation) requires maintaining the activations across the entire dataset in memory, which is challenging for wide and deep networks [3].

Therefore, we need to define the unbiased estimator of HSIC so that the value of CKA is independent of the batch size

$$HSIC_1(K, L)=\frac{1}{n(n-3)}\left( tr(\tilde{K}, \tilde{L}) + \frac{1^T\tilde{K}11^T\tilde{L}1}{(n-1)(n-2)} - \frac{2}{n-2}1^T\tilde{K}\tilde{L}1\right),$$

where $\tilde{K}$ and $\tilde{L}$ are obtained by setting the diagonal entries of $K$ and $L$ to zero. Finally, we can compute the minibatch version of CKA by averaging $HSIC_1$ scores over $k$ minibatches

$$CKA_{minibatch}=\frac{\frac{1}{k} \displaystyle\sum_{i=1}^{k} HSIC_1(K_i, L_i)}{\sqrt{\frac{1}{k} \displaystyle\sum_{i=1}^{k} HSIC_1(K_i, K_i)}\sqrt{\frac{1}{k} \displaystyle\sum_{i=1}^{k} HSIC_1(L_i, L_i)}},$$

with $K_i=X_iX_i^T$ and $L_i=Y_iY_i^T$, where $X_i \in \mathbb{R}^{m \times p_1}$ and $Y_i \in \mathbb{R}^{m \times p_2}$ are now matrices containing activations of the $i^{th}$ minibatch of $m$ examples sampled without replacement [3].

---

## 📦 Installation
This project requires python >= 3.10.

### Create a new venv
```bash
# If you have uv installed
uv venv

# Otherwise
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # if you are on Linux
.\.venv\Scripts\activate.bat  # if you are using the cmd on Windows
.\.venv\Scripts\Activate.ps1  # if you are using the PowerShell on Windows
```

### Install the package
> [!NOTE]
> This will install <img height="15" width="15" src="https://cdn.simpleicons.org/pytorch"/>PyTorch compiled with CUDA.

You can install the package:
- _from PyPI_
  ```bash
  # Using uv
  uv pip install ckatorch

  # Using pip
  pip install ckatorch
  ```

- _from this repo_
  ```bash
  # Using uv
  uv pip install git+https://github.com/RistoAle97/centered-kernel-alignment

  # Using pip
  pip install git+https://github.com/RistoAle97/centered-kernel-alignment
  ```

- _by cloning the repository and installing the dependencies_
  ```bash
  git clone https://github.com/RistoAle97/centered-kernel-alignment

  # If you have uv installed
  uv pip install -e centered-kernel-alignment
  uv pip install ckatorch --group dev  # if you want to also install the dev dependencies

  # Otherwise
  pip install -e centered-kernel-alignment
  pip install ckatorch --group dev # same as for uv, remember to open a pull request afterwards
  ```

Take a look at the `examples` directory to understand how to compute CKA in two basic scenarios.

---

## 🖼️	Plots
> [!NOTE]
> The comparison makes more sense if the models share a common architecture.

Model compared with itself             |  Different models compared
:-------------------------:|:-------------------------:
![Model compared with itself](https://raw.githubusercontent.com/RistoAle97/centered-kernel-alignment/refs/heads/main/plots/model_comparison_itself.png)  |  ![Model comparison](https://raw.githubusercontent.com/RistoAle97/centered-kernel-alignment/refs/heads/main/plots/model_comparison.png)

---

## 📚 Bibliography
[1] Kornblith, Simon, et al. ["Similarity of neural network representations revisited."](https://arxiv.org/abs/1905.00414) *International Conference on Machine Learning*. PMLR, 2019.

[2] Wang, Tinghua, Xiaolu Dai, and Yuze Liu. ["Learning with Hilbert–Schmidt independence criterion: A review and new perspectives."](https://www.sciencedirect.com/science/article/pii/S0950705121008297) *Knowledge-based systems* 234 (2021): 107567.

[3] Nguyen, Thao, Maithra Raghu, and Simon Kornblith. ["Do wide and deep networks learn the same things? uncovering how neural network representations vary with width and depth."](https://arxiv.org/abs/2010.15327) *arXiv preprint* arXiv:2010.15327 (2020).

This project is also based on the following repositories:
- [representation_similarity](https://github.com/google-research/google-research/tree/master/representation_similarity) (original implementation).
- [PyTorch-Model-Compare](https://github.com/AntixK/PyTorch-Model-Compare) (nice PyTorch implementation that employs hooks).
- [CKA.pytorch](https://github.com/numpee/CKA.pytorch) (minibatch version of CKA and useful batched implementation of $HSIC_1$).

---

## 📝 License
This project is [MIT licensed](https://github.com/RistoAle97/centered-kernel-alignment/blob/main/LICENSE).
