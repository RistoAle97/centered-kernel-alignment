<div align="center">

# :hammer_and_wrench: CKA PyTorch examples :hammer_and_wrench:
**Some examples of how to use the package.**

---
</div>

## :pushpin: Examples
> [!NOTE]
> It is recommended to use hooks instead of feature extractors, as the latter will need some workarounds to work smoothly on complex architectures. For that, you just need to set `use_hooks=True` while initializing a CKA object.

- Comparison between two models, each consisting of a stack of feed-forward networks. This is the simplest example and should be used as a starting point.
- Comparison between two randomly initialized Bert models using the HuggingFace libraries. This examples uses hooks instead of feature extractors.
