<div align="center">

# :hammer_and_wrench: CKA PyTorch examples :hammer_and_wrench:
**Some examples of how to use the package.**

---
</div>

## :hammer_and_wrench: Examples
> [!CAUTION]
> Currently, HuggingFace models will work with this package since `create_feature_extractor` needs a very specific parameter (`concrete_args`) that will be added in the next release of torchvision (0.1.9 or 0.2.0).
> If you are working with a custom transformer architecture you should, at least, set `first_leaf_modules=[YourPositionalEncoderClass]` and `second_leaf_modules=[YourPositinalEncoderClass]` to avoid throwing an exception.

- Comparison between two networks, each consisting of a stack of feed-forward networks. This is the simplest example and should be used as a starting point.
- **(Not complete)** Comparison between two randomly initialized Bert models using the HuggingFace libraries. This example gives a deeper overview of the package.