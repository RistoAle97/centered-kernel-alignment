import inspect
from collections.abc import Callable
from functools import partial
from typing import Any
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from transformers import PreTrainedModel

from .core import cka_batch


class CKA:
    def __init__(
        self,
        first_model: nn.Module,
        second_model: nn.Module,
        layers: list[str],
        second_layers: list[str] | None = None,
        first_leaf_modules: list[type[nn.Module]] | None = None,
        second_leaf_modules: list[type[nn.Module]] | None = None,
        first_name: str | None = None,
        second_name: str | None = None,
        use_hooks: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        """Centered Kernel Alignment (CKA) implementation. Given a set of examples, CKA compares the representations of
        examples passed through the layers that we want to compare.
        :param first_model: the first model whose layer features we want to compare.
        :param second_model: the second model whose layer features we want to compare.
        :param layers: list of layers name under inspection (if no "second_layers" is provided, then the layers will
            count for the second model too).
        :param second_layers: list of layers from the second model under inspection (default=None).
        :param first_leaf_modules: list of problematic layers that will not be traced by the first extractor
            (default=None).
        :param second_leaf_modules: list of problematic layers that will not be traced by the second extractor
            (default=None).
        :param first_name: name of the first model (default=None).
        :param second_name: name of the second model (default=None).
        :param use_hooks: whether to use hooks instead of the feature extractors. This parameter will be forcibly set as
            True in case you are working with a HuggingFace model(default=False).
        :param device: the device used during the computation (default="cpu").
        """
        # Set up the device
        self.device = torch.device(device)

        # Check if no layers were passed
        if layers is None or len(layers) == 0:
            raise ValueError(
                "You can not pass 'None' or an empty list as layers. We suggest using 'first_model.named_modules()'"
                "in order to see which layers can be passed."
            )

        # Remove potential duplicates
        layers = sorted(set(layers), key=layers.index)

        # Check if to many layers were passed
        if len(layers) > 100:
            warn(
                f"You passed {len(layers)} distinct layers, which is way too high. Consider passing only those"
                f"layers whose features you are really interested about."
            )

        # Copy the first model's layers if they are not passed
        if second_layers is None or len(second_layers) == 0:
            second_layers = layers.copy()
        else:
            # Remove potential duplicates
            second_layers = sorted(set(second_layers), key=second_layers.index)

            # Check if too many layers were passed
            if len(second_layers) > 100:
                warn(
                    f"You passed {len(second_layers)} distinct layers for the second model, which is way too high."
                    f"Consider passing only those layers whose features you are really interested about."
                )

        # Dicts where the output of each layer (i.e.: the features) will be saved while using hooks
        self.first_features: dict[str, torch.Tensor] = {}
        self.second_features: dict[str, torch.Tensor] = {}

        # The CKA computation can be performed through hooks or feature extractors, the results will be the same
        self.use_hooks = use_hooks
        if use_hooks or isinstance(first_model, PreTrainedModel) or isinstance(second_model, PreTrainedModel):
            # Insert a hook for each layer
            layers, second_layers = self._insert_hooks(first_model, second_model, layers, second_layers)
            self.first_model = first_model.to(device)
            self.second_model = second_model.to(device)
        else:
            # Deal with the non-traceable layers
            first_tracer_kwargs = {"leaf_modules": first_leaf_modules} if first_leaf_modules is not None else None
            second_tracer_kwargs = {"leaf_modules": second_leaf_modules} if second_leaf_modules is not None else None

            # Build the extractors, they work like a normal torch.nn.Module, but their output is a dict containing the
            # features of each layer under their scope.
            self.first_extractor = create_feature_extractor(
                model=first_model,
                return_nodes=layers,
                tracer_kwargs=first_tracer_kwargs,
            ).to(self.device)
            self.second_extractor = create_feature_extractor(
                model=second_model,
                return_nodes=second_layers,
                tracer_kwargs=second_tracer_kwargs,
            ).to(self.device)

        # Manage the models names
        first_name = first_name if first_name is not None else first_model.__repr__().split("(")[0]
        if first_model is second_model:
            second_name = first_name
        else:
            second_name = second_name if second_name is not None else second_model.__repr__().split("(")[0]
            if first_name == second_name:
                warn(f"Both models are called {first_name}. This may cause confusion when analyzing the results.")

        # Set up the models infos
        self.first_model_infos = {"name": first_name, "layers": layers}
        self.second_model_infos = {"name": second_name, "layers": second_layers}

    def _hook(self, model: str, module_name: str, module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:
        del module, inp  # delete unused arguments so that we can pass the linter checks
        if model == "first":
            self.first_features[module_name] = out.detach()
        else:
            self.second_features[module_name] = out.detach()

    def _insert_hooks(
        self,
        first_model: nn.Module,
        second_model: nn.Module,
        layers: list[str],
        second_layers: list[str],
    ) -> tuple[list[str], list[str]]:
        # Only those layers that were found will be placed inside the following lists
        filtered_layers = []
        filtered_second_layers = []

        # Add hooks for the first model's layers
        for module_name, module in list(first_model.named_modules()):
            if module_name in layers:
                module.register_forward_hook(partial(self._hook, "first", module_name))
                filtered_layers.append(module_name)

        # Add hooks for the second model's layers
        for module_name, module in list(second_model.named_modules()):
            if module_name in second_layers:
                module.register_forward_hook(partial(self._hook, "second", module_name))
                filtered_second_layers.append(module_name)

        # One last check
        if len(filtered_layers) == 0 or len(filtered_second_layers) == 0:
            raise ValueError(
                "No layers were found for one of the two models, please use 'model.named_modules()' in order to check"
                "which layers can be passed to the method."
            )

        return filtered_layers, filtered_second_layers

    def __call__(
        self,
        dataloader: DataLoader,
        epochs: int = 10,
        f_extract: Callable[..., dict[str, torch.Tensor]] | None = None,
        f_args: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Process inputs and computes the CKA matrix. Note that this computation uses the minibatch version of CKA by
        Nguyen et al. (https://arxiv.org/abs/2010.15327).
        :param dataloader: dataloader that will be used during the computation.
        :param epochs: number of iterations over the dataloader (default=10).
        :param f_extract: the function to apply on the dataloader, this function should take any number and type of
            inputs and return a dict. If no function is passed, then some checks will be applied for finding the actual
            type of the batch (default=None).
        :param f_args: the arguments passed to the f_extract function (default=None).
        :return: the CKA value.
        """
        if dataloader.drop_last:
            raise ValueError(
                "The argument 'drop_last' must be set to False otherwise you will get very different values by varying"
                "the batch size."
            )

        if not isinstance(dataloader.sampler, RandomSampler):
            warn("We suggest setting 'shuffle=True' in your dataloader in order to have a less biased computation.")

        if self.use_hooks:
            self.first_model.eval()
            self.second_model.eval()
        else:
            self.first_extractor.eval()
            self.second_extractor.eval()

        with torch.no_grad():
            n = len(self.first_model_infos["layers"])
            m = len(self.second_model_infos["layers"])

            # Iterate through the dataset
            num_batches = len(dataloader)
            cka_matrices = []
            for epoch in tqdm(range(epochs), desc="| Computing CKA |", total=epochs):
                cka_epoch = torch.zeros(n, m, device=self.device)
                for batch in tqdm(dataloader, desc=f"| Computing CKA epoch {epoch} |", total=num_batches, leave=False):
                    self.first_features = {}
                    self.second_features = {}
                    if f_extract is not None:
                        # Apply the provided function and put everything on the device
                        f_extract = {} if f_extract is None else f_extract
                        batch = f_extract(batch, **f_args)
                        batch = {f"{name}": batch_input.to(self.device) for name, batch_input in batch.items()}
                    elif isinstance(batch, list | tuple):
                        arg_method = self.first_model.forward if self.use_hooks else self.first_extractor.forward
                        args_list = inspect.getfullargspec(arg_method).args[1:]  # skip "self" arg
                        batch = {f"{args_list[i]}": batch_input.to(self.device) for i, batch_input in enumerate(batch)}
                    elif not isinstance(batch, dict):
                        raise ValueError(
                            f"Type {type(batch)} is not supported for the CKA computation. We suggest building a custom"
                            f"'Dataset' class such that the '__get_item__' method returns a dict[str, Any]."
                        )

                    # Do a forward pass for both models
                    if self.use_hooks:
                        _ = self.first_model(**batch)
                        _ = self.second_model(**batch)
                        first_outputs = self.first_features
                        second_outputs = self.second_features
                    else:
                        first_outputs = self.first_extractor(**batch)
                        second_outputs = self.second_extractor(**batch)

                    # Compute the CKA values for each output pair
                    for i, (_, x) in enumerate(first_outputs.items()):
                        for j, (_, y) in enumerate(second_outputs.items()):
                            cka_epoch[i, j] = cka_batch(x, y)

                cka_matrices.append(cka_epoch)

        cka = torch.stack(cka_matrices)
        cka = cka.sum(0) / epochs

        # One last check
        if torch.isnan(cka).any():
            raise ValueError("CKA computation resulted in NANs.")

        return cka

    def plot_cka(
        self,
        cka_matrix: torch.Tensor,
        save_path: str | None = None,
        title: str | None = None,
        show_ticks_labels: bool = False,
        short_tick_labels_splits: int | None = None,
        use_tight_layout: bool = True,
        show_annotations: bool = True,
        show_img: bool = True,
        show_half_heatmap: bool = False,
        **kwargs,
    ) -> None:
        """Plot the CKA matrix obtained calling this class' forward() method.
        :param cka_matrix: the CKA matrix.
        :param save_path: the path where to save the plot, if None then the plot will not be saved (default=None).
        :param title: the plot title, if None then no title will be used (default=None).
        :param show_ticks_labels: whether to show the tick labels (default=False).
        :param short_tick_labels_splits: only works when show_tick_labels is True. If it is not None, the tick labels
            will be shortened to the defined sublayer starting from the deepest level. E.g.: if the layer name is
            'encoder.ff.linear' and this parameter is set to 1, then only 'linear' will be printed on the heatmap
            (default=None).
        :param use_tight_layout: whether to use a tight layout in order not to cut any label in the plot (default=True).
        :param show_annotations: whether to show the annotations on the heatmap (default=True).
        :param show_img: whether to show the plot (default=True).
        :param show_half_heatmap: whether to mask the upper left part of the heatmap since those valued are duplicates
            (default=False).
        """
        # Deal with some arguments
        vmin: float | None = kwargs.get("vmin", None)
        vmax: float | None = kwargs.get("vmax", None)
        if (vmin is not None) ^ (vmax is not None):
            raise ValueError("'vmin' and 'vmax' must be defined together or both equal to None.")

        cmap = kwargs.get("cmap", "magma")
        vmin = min(vmin, torch.min(cka_matrix).item()) if vmin is not None else vmin
        vmax = max(vmax, torch.max(cka_matrix).item()) if vmax is not None else vmax
        mask = np.tril(np.ones_like(cka_matrix.cpu(), dtype=bool), k=-1) if show_half_heatmap else None

        # Build the heatmap
        ax = sn.heatmap(cka_matrix.cpu(), vmin=vmin, vmax=vmax, annot=show_annotations, cmap=cmap, mask=mask)
        ax.invert_yaxis()
        ax.set_xlabel(f"{self.second_model_infos['name']} layers", fontsize=12)
        ax.set_ylabel(f"{self.first_model_infos['name']} layers", fontsize=12)

        # Deal with tick labels
        if show_ticks_labels:
            if short_tick_labels_splits is None:
                ax.set_xticklabels(self.second_model_infos["name"])
                ax.set_yticklabels(self.first_model_infos["name"])
            else:
                ax.set_xticklabels(
                    [
                        "-".join(module.split(".")[-short_tick_labels_splits:])
                        for module in self.second_model_infos["layers"]
                    ]
                )
                ax.set_yticklabels(
                    [
                        "-".join(module.split(".")[-short_tick_labels_splits:])
                        for module in self.first_model_infos["layers"]
                    ]
                )

            plt.xticks(rotation=90)
            plt.yticks(rotation=0)

        # Put the title if passed
        if title is not None:
            ax.set_title(title, fontsize=14)
        else:
            title = f"{self.first_model_infos['name']} vs {self.second_model_infos['name']}"
            ax.set_title(title, fontsize=14)

        # Set the layout to tight if the corresponding parameter is True
        if use_tight_layout:
            plt.tight_layout()

        # Save the plot to the specified path if defined
        if save_path is not None:
            title = title.replace("/", "-")
            path_rel = f"{save_path}/{title}.png"
            plt.savefig(path_rel, dpi=400, bbox_inches="tight")

        # Show the image if the user chooses to do so
        if show_img:
            plt.show()
