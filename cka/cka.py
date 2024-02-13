import inspect
from typing import Callable, Literal
from warnings import warn

import matplotlib.pyplot as plt
import seaborn as sn
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from .utils import linear_kernel, rbf_kernel, center_matrix


class CKA(nn.Module):

    def __init__(
        self,
        first_model: nn.Module,
        second_model: nn.Module,
        layers: list[str] | set[str],
        second_layers: list[str] | set[str] = None,
        first_name: str = None,
        second_name: str = None,
        device: str = "cpu",
        kernel: Literal["linear", "rbf"] = "linear",
    ) -> None:
        """
        Centered Kernel Alignment (CKA) implementation. Given a set of examples, CKA compares the representations of
        examples passed through the layers that we want to compare.
        :param first_model: the first model whose layer features we want to compare.
        :param second_model: the second model whose layer features we want to compare.
        :param layers: list of layers name under inspection (if no "second_layers" is provided, then the layers will
            count for the second model too).
        :param second_layers: list of layers from the second model under inspection (default=None).
        :param first_name: name of the first model (default=None).
        :param second_name: name of the second model (default=None).
        :param device: the device used during the computation (default="cpu").
        :param kernel: the type of kernel, can be either "linear" or "rbf" (default="linear").
        """
        super().__init__()

        # Set up the kernel
        assert kernel in ["rbf", "linear"], ValueError("The kernel must be either 'linear' or 'rbf'.")
        self.kernel = kernel

        # Check if no layers were passed and if there are too many of them
        layers = set(layers)
        assert layers is not None and len(layers) > 0, ValueError(
            "You can not pass 'None' or an empty list as layers. We suggest using 'get_graph_node_names' from the"
            "'torchvision' package in order to see which layers can be passed."
        )
        if len(layers) > 100:
            warn(
                f"You passed {len(layers)} distinct layers, which is way too high. Consider passing only those"
                f"layers whose features you are really interested about."
            )
        if second_layers is None or len(second_layers) == 0:
            second_layers = layers.copy()
        else:
            second_layers = set(second_layers)
            if len(second_layers) > 100:
                warn(
                    f"You passed {len(second_layers)} distinct layers for the second model, which is way too high."
                    f"Consider passing only those layers whose features you are really interested about."
                )

        # Build the extractors, they work like a normal torch.nn.Module, but their output is a dict containing the
        # features of each layer under their scope.
        self.first_extractor = create_feature_extractor(first_model, list(layers)).to(device)
        self.second_extractor = create_feature_extractor(second_model, list(second_layers)).to(device)

        # Manage the models names
        first_name = first_name if first_name is not None else first_model.__repr__().split("(")[0]
        second_name = second_name if second_name is not None else second_model.__repr__().split("(")[0]
        if first_name == second_name:
            warn(f"Both models are called {first_name}, beware that it may cause confusion when analyzing the results.")

        # Set up the models infos
        self.first_model_infos = {"name": first_name, "layers": layers}
        self.second_model_infos = {"name": second_name, "layers": second_layers}

        # At last, save the device used during the computation
        self.device = torch.device(device)

    def compute_cka(
        self, x: torch.Tensor, y: torch.Tensor, unbiased: bool = False, rbf_threshold: float = 1.0
    ) -> torch.Tensor:
        """
        Compute the Centered Kernel Alignment between two given matrices. Adapted from the one made by Kornblith et al.
        https://github.com/google-research/google-research/tree/master/representation_similarity.
        :param x: tensor of shape (n, j).
        :param y: tensor of shape (n, k).
        :param unbiased: whether to use the unbiased version of CKA (default=False).
        :param rbf_threshold: the threshold used by the RBF kernel (default=1.0).
        :return: a float in [0, 1] that is the CKA value between the two given matrices.
        """
        # We need to change the dtype of the tensors for a better precision
        x = x.type(torch.float64) if not x.dtype == torch.float64 else x
        y = y.type(torch.float64) if not x.dtype == torch.float64 else y

        # Apply the kernel to the matrices and center them
        if self.kernel == "linear":
            f_kernel = linear_kernel
            kernel_other_args = {}
        else:
            f_kernel = rbf_kernel
            kernel_other_args = {"threshold": rbf_threshold}

        gram_x = f_kernel(x, **kernel_other_args)
        gram_y = f_kernel(y, **kernel_other_args)
        centered_gram_x = center_matrix(gram_x, unbiased)
        centered_gram_y = center_matrix(gram_y, unbiased)

        # This is the final step of computing the Frobenius norm of Y^T * X
        hsic_xy = centered_gram_x.ravel().dot(centered_gram_y.ravel())

        # Compute the Frobenius norm for both matrix
        fro_norm_x = torch.linalg.norm(centered_gram_x, ord="fro")
        fro_norm_y = torch.linalg.norm(centered_gram_y, ord="fro")

        # Finally, compute CKA
        cka = hsic_xy / (fro_norm_x * fro_norm_y)
        return cka

    def forward(
        self,
        dataloader: DataLoader,
        unbiased: bool = False,
        rbf_threshold: float = 1.0,
        f_extract: Callable[..., dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """

        :param dataloader:
        :param unbiased:
        :param rbf_threshold:
        :param f_extract: the function to apply on the dataloader, this function should take any number and type of
            inputs and return a dict. If no function is passed, then some checks will be applied for finding the actual
            type of the batch (default=None).
        :return:
        """
        self.first_extractor.eval()
        self.second_extractor.eval()

        with torch.no_grad():
            n = len(self.first_model_infos["layers"])
            m = len(self.second_model_infos["layers"])
            cka = torch.zeros(n, m, device=self.device)

            # Iterate through the dataset
            num_batches = len(dataloader)
            for batch in tqdm(dataloader, desc="| Computing CKA |", total=num_batches):
                if f_extract is not None:
                    # Apply the provided function and put everything on the device
                    batch: dict[str, torch.Tensor] = f_extract(batch)
                    batch = {f"{name}": batch_input.to(self.device) for name, batch_input in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    args_list = inspect.getfullargspec(self.first_extractor.forward).args[1:]  # skip "self" argument
                    batch = {f"{args_list[i]}": batch_input.to(self.device) for i, batch_input in enumerate(batch)}
                else:
                    raise ValueError(
                        f"Type {type(batch)} is not supported for the CKA computation. We suggest building a custom"
                        f"'Dataset' class such that the '__get_item__' method returns a dict[str, torch.Tensor]."
                    )

                # Do a forward pass for both models
                first_outputs: dict[str, torch.Tensor] = self.first_extractor(**batch)
                second_outputs: dict[str, torch.Tensor] = self.second_extractor(**batch)
                for i, (first_out_name, first_out) in enumerate(first_outputs.items()):
                    x = first_out.view(-1, first_out.shape[-1])
                    for j, (second_out_name, second_out) in enumerate(second_outputs.items()):
                        y = second_out.view(-1, second_out.shape[-1])
                        cka[i, j] += self.compute_cka(x, y, unbiased, rbf_threshold) / num_batches

        # One last check
        assert not torch.isnan(cka).any(), "CKA computation resulted in NANs"

        return cka

    def plot_results(
        self,
        cka_matrix: torch.Tensor,
        save_path: str = None,
        title: str = None,
        show_ticks_labels: bool = False,
        short_tick_labels_splits: int | None = None,
        use_tight_layout: bool = True,
        show_annotations: bool = True,
        show_img: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot the CKA matrix obtained calling this class' forward() method.
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
        :return:
        """
        # Build the heatmap
        vmin: float | None = kwargs.get("vmin", None)
        vmax: float | None = kwargs.get("vmax", None)
        if (vmin is not None) ^ (vmax is not None):
            raise ValueError("'vmin' and 'vmax' must be defined together or both equal to None.")

        cmap = kwargs.get("cmap", "magma")
        vmin = min(vmin, torch.min(cka_matrix).item()) if vmin is not None else vmin
        vmax = max(vmax, torch.max(cka_matrix).item()) if vmax is not None else vmax
        ax = sn.heatmap(cka_matrix.cpu(), vmin=vmin, vmax=vmax, annot=show_annotations, cmap=cmap)
        ax.invert_yaxis()
        ax.set_xlabel(f"Layers {self.second_model_infos['name']}", fontsize=10)
        ax.set_ylabel(f"Layers {self.first_model_infos['name']}", fontsize=10)

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
        chart_title = title
        if title is not None:
            ax.set_title(f"{title}", fontsize=10)
        else:
            chart_title = f"{self.first_model_infos['name']} vs {self.second_model_infos['name']}"
            ax.set_title(chart_title, fontsize=10)

        # Set the layout to tight if the corresponding parameter is True
        if use_tight_layout:
            plt.tight_layout()

        # Save the plot to the specified path if defined
        if save_path is not None:
            chart_title = chart_title.replace("/", "-")
            path_rel = f"{save_path}/{chart_title}.png"
            plt.savefig(path_rel, dpi=400, bbox_inches="tight")

        # Show the image if the user chooses to do so
        if show_img:
            plt.show()
