"""Module for computing CKA in order to compare two PyTorch models through their layers' activations."""

import inspect
import json
import random
import re
import string
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal
from warnings import warn

import torch
import yaml
from safetensors.torch import save_file, save_model
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from .core import cka_batch
from .plot import plot_cka


@dataclass
class ModelInfo:
    """Class for storing the model info."""

    name: str
    layers: list[str]


class CKA:
    """Centered Kernel Alignment (CKA) implementation.

    CKA is a similarity index between representations of features in neural networks, based on the
    Hilbert-Schmidt Independence Criterion (HSIC). Given a set of examples, CKA compares the representations of examples
    passed through the layers that we want to compare.
    """

    def __init__(
        self,
        first_model: nn.Module,
        second_model: nn.Module,
        layers: list[str],
        second_layers: list[str] | None = None,
        first_name: str | None = None,
        second_name: str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initializes a CKA object.

        Args:
            first_model (torch.nn.Module): the first model whose layer features we want to compare.
            second_model (torch.nn.Module): the second model whose layer features we want to compare.
            layers (list[str]): list of layers name under inspection (if no "second_layers" is provided, then the
                layers will count for the second model too).
            second_layers (list[str] | None): list of layers from the second model under inspection (default=None).
            first_name (str | None): name of the first model (default=None).
            second_name (str | None): name of the second model (default=None).
            device (str | torch.device): the device used during the computation (default="cpu").

        Raises:
            ValueError: if ``layers`` is None or empty.
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
                f"layers whose features you are really interested about.",
                stacklevel=2,
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
                    f"Consider passing only those layers whose features you are really interested about.",
                    stacklevel=2,
                )

        # Dicts where the output of each layer (i.e.: the features) will be saved while using hooks
        self.first_features: dict[str, torch.Tensor] = {}
        self.second_features: dict[str, torch.Tensor] = {}

        # Insert a hook for each layer
        layers, second_layers = self._insert_hooks(first_model, second_model, layers, second_layers)
        self.first_model = first_model.to(device)
        self.second_model = second_model.to(device)

        # Manage the models names
        first_name = first_name if first_name is not None else first_model.__repr__().split("(")[0]
        if first_model is second_model:
            second_name = first_name
        else:
            second_name = second_name if second_name is not None else second_model.__repr__().split("(")[0]
            if first_name == second_name:
                warn(
                    f"Both models are called {first_name}. This may cause confusion when analyzing the results.",
                    stacklevel=2,
                )

        # Set up the models info
        first_name = re.sub("[^0-9a-zA-Z_]+", "", first_name.replace(" ", "_"))
        second_name = re.sub("[^0-9a-zA-Z_]+", "", second_name.replace(" ", "_"))
        self.first_model_info = ModelInfo(first_name, layers)
        self.second_model_info = ModelInfo(second_name, second_layers)

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
        filtered_first_layers = []
        filtered_second_layers = []

        # Add hooks for the first model's layers
        for module_name, module in list(first_model.named_modules()):
            if module_name in layers:
                module.register_forward_hook(partial(self._hook, "first", module_name))
                filtered_first_layers.append(module_name)

        # Add hooks for the second model's layers
        for module_name, module in list(second_model.named_modules()):
            if module_name in second_layers:
                module.register_forward_hook(partial(self._hook, "second", module_name))
                filtered_second_layers.append(module_name)

        # One last check
        if len(filtered_first_layers) == 0 or len(filtered_second_layers) == 0:
            raise ValueError(
                "No layers were found for one of the two models, please use 'model.named_modules()' in order to check"
                "which layers can be passed to the method."
            )

        return filtered_first_layers, filtered_second_layers

    def __call__(
        self,
        dataloader: DataLoader,
        epochs: int = 10,
        f_extract: Callable[..., dict[str, torch.Tensor]] | None = None,
        f_args: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Process inputs and computes the CKA matrix.

        This computation employs the minibatch version of CKA by Nguyen et al. (https://arxiv.org/abs/2010.15327).

        Args:
            dataloader (torch.utils.data.Dataloader): dataloader that will be used during the computation.
            epochs (int): number of iterations over the dataloader (default=10).
            f_extract (Callable[..., dict[str, torch.Tensor]] | None): the function to apply on the dataloader, this
                function should take any number and type of inputs and return a dict. If no function is passed, then
                some checks will be applied for finding the actual type of the batch (default=None).
            f_args (dict[str, Any] | None): the arguments passed to the f_extract function (default=None).

        Returns:
            torch.Tensor: a tensor representing the CKA matrix.

        Raises:
            ValueError: if the parameter 'drop_last' of the dataloader is set to True or if the batch type is not
                supported.
        """
        if dataloader.drop_last:
            raise ValueError(
                "The argument 'drop_last' must be set to False otherwise you will get very different values by varying"
                "the batch size."
            )

        if not isinstance(dataloader.sampler, RandomSampler):
            warn(
                "We suggest setting 'shuffle=True' in your dataloader in order to have a less biased computation.",
                stacklevel=2,
            )

        self.first_model.eval()
        self.second_model.eval()

        with torch.no_grad():
            n = len(self.first_model_info.layers)
            m = len(self.second_model_info.layers)

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
                        arg_method = self.first_model.forward
                        args_list = inspect.getfullargspec(arg_method).args[1:]  # skip "self" arg
                        batch = {f"{args_list[i]}": batch_input.to(self.device) for i, batch_input in enumerate(batch)}
                    elif not isinstance(batch, dict):
                        raise ValueError(
                            f"Type {type(batch)} is not supported for the CKA computation. We suggest building a custom"
                            f"'Dataset' class such that the '__get_item__' method returns a dict[str, Any]."
                        )

                    # Do a forward pass for both models
                    _ = self.first_model(**batch)
                    _ = self.second_model(**batch)
                    first_outputs = self.first_features
                    second_outputs = self.second_features

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
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "magma",
        show_ticks_labels: bool = False,
        short_tick_labels_splits: int | None = None,
        use_tight_layout: bool = True,
        show_annotations: bool = True,
        show_img: bool = True,
        show_half_heatmap: bool = False,
        invert_y_axis: bool = True,
    ) -> None:
        """Plot the CKA matrix obtained by calling this class' __call__() method.

        Args:
            cka_matrix (torch.Tensor): the CKA matrix.
            save_path (str | None): where to save the plot, if None then the plot will not be saved (default=None).
            title (str | None): the plot title, if None then a simple text with the name of both models will be used
                (default=None).
            vmin (float | None): values to anchor the colormap, otherwise they are inferred from the data and other
                keyword arguments (default=None).
            vmax (float | None): values to anchor the colormap, otherwise they are inferred from the data and other
                keyword arguments (default=None).
            cmap (str): the name of the colormap to use (default="magma").
            show_ticks_labels (bool): whether to show the tick labels (default=False).
            short_tick_labels_splits (int | None): only works when show_tick_labels is True. If it is not None, the
                tick labels will be shortened to the defined sublayer starting from the deepest level. E.g.: if the
                layer name is 'encoder.ff.linear' and this parameter is set to 1, then only 'linear' will be printed on
                the heatmap (default=None).
            use_tight_layout (bool): whether to use a tight layout in order not to cut any label in the plot
                (default=True).
            show_annotations (bool): whether to show the annotations on the heatmap (default=True).
            show_img (bool): whether to show the plot (default=True).
            show_half_heatmap (bool): whether to mask the upper left part of the heatmap since those valued are
                duplicates (default=False).
            invert_y_axis (bool): whether to invert the y-axis of the plot (default=True).


        Raises:
            ValueError: if ``vmax`` or ``vmin`` are not defined together or both equal to None.
        """
        plot_cka(
            cka_matrix=cka_matrix,
            first_layers=self.first_model_info.layers,
            second_layers=self.second_model_info.layers,
            first_name=self.first_model_info.name,
            second_name=self.second_model_info.name,
            save_path=save_path,
            title=title,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            show_ticks_labels=show_ticks_labels,
            short_tick_labels_splits=short_tick_labels_splits,
            use_tight_layout=use_tight_layout,
            show_annotations=show_annotations,
            show_img=show_img,
            show_half_heatmap=show_half_heatmap,
            invert_y_axis=invert_y_axis,
        )

    def save(
        self,
        cka_matrix: torch.Tensor,
        dir_path: str | Path = "",
        file_format: Literal["json", "yaml"] = "json",
        save_models: bool = False,
    ) -> None:
        """Saves the CKA matrix and the info about the models used during the computation.

        Note that the CKA matrix and the models' weights are saved as safetensors files and their paths are be stored
        inside the json or yaml file.

        Args:
            cka_matrix (torch.Tensor): the CKA matrix.
            dir_path (str | pathlib.Path): where to save the weights of the models and the file with their info
                (default="").
            file_format (Literal["json", "yaml"]): in which format the output is saved, can be either 'json' or 'yaml'
                (default: "json").
            save_models (bool): whether to also save the models used for the CKA computation (default=False).

        Raises:
            ValueError: if ``file_format`` not in ['json', 'yaml'].
        """
        # Save the CKA matrix
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        cka_tensor_path = dir_path / "cka.safetensors"
        cka_info_path = dir_path / f"cka.{file_format}"
        random_str = ""
        if cka_tensor_path.exists():
            # If the file already exists, change the name by appending five random characters
            random_str = "_" + "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))
            cka_tensor_path = dir_path / f"cka{random_str}.safetensors"
            cka_info_path = dir_path / f"cka{random_str}.{file_format}"

        save_file({"cka": cka_matrix}, cka_tensor_path)

        # Build the dict to dump
        cka_info = {
            "cka_matrix": {
                "path": str(cka_tensor_path.absolute()),
            },
            "first_model": asdict(self.first_model_info),
            "second_model": asdict(self.second_model_info),
        }

        # Save the models if requested
        if save_models:
            first_model_path = dir_path / f"{self.first_model_info.name}{random_str}.safetensors"
            second_model_path = dir_path / f"{self.second_model_info.name}{random_str}.safetensors"
            save_model(self.first_model, first_model_path)
            save_model(self.second_model, second_model_path)
            cka_info["first_model"]["path"] = str(first_model_path.absolute())
            cka_info["second_model"]["path"] = str(second_model_path.absolute())

        # Dump the dict
        with open(cka_info_path, "w", encoding="utf-8") as dump_file:
            match file_format:
                case "json":
                    json.dump(cka_info, dump_file, indent=4)
                case "yaml":
                    yaml.dump(cka_info, dump_file, indent=4)
                case _:
                    raise ValueError(f"Unsupported format '{file_format}'")
