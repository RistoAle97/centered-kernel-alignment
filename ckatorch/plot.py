"""Utilities for plotting the CKA matrices."""

import matplotlib.pyplot as plt
import seaborn as sn
import torch


def plot_cka(
    cka_matrix: torch.Tensor,
    first_layers: list[str],
    second_layers: list[str],
    first_name: str = "First Model",
    second_name: str = "Second Model",
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
    """Plot the CKA matrix obtained by calling CKA class __call__() method.

    Args:
        cka_matrix (torch.Tensor): the CKA matrix.
        first_layers (list[str]): list of the names of the first model's layers.
        second_layers (list[str]): list of the names of the second model's layers.
        first_name (str): name of the first model (default="First Model").
        second_name (str): name of the second model (default="Second Model").
        save_path (str | None): where to save the plot, if None then the plot will not be saved (default=None).
        title (str | None): the plot title, if None then a simple text with the name of both models will be used
            (default=None).
        vmin (float | None): values to anchor the colormap, otherwise they are inferred from the data and other keyword
            arguments (default=None).
        vmax (float | None): values to anchor the colormap, otherwise they are inferred from the data and other keyword
            arguments (default=None).
        cmap (str): the name of the colormap to use (default="magma").
        show_ticks_labels (bool): whether to show the tick labels (default=False).
        short_tick_labels_splits (int | None): only works when show_tick_labels is True. If it is not None, the tick
            labels will be shortened to the defined sublayer starting from the deepest level. E.g.: if the layer name
            is 'encoder.ff.linear' and this parameter is set to 1, then only 'linear' will be printed on the heatmap
            (default=None).
        use_tight_layout (bool): whether to use a tight layout in order not to cut any label in the plot (default=True).
        show_annotations (bool): whether to show the annotations on the heatmap (default=True).
        show_img (bool): whether to show the plot (default=True).
        show_half_heatmap (bool): whether to mask the upper left part of the heatmap since those valued are duplicates
            (default=False).
        invert_y_axis (bool): whether to invert the y-axis of the plot (default=True).

    Raises:
        ValueError: if ``vmax`` or ``vmin`` are not defined together or both equal to None.
    """
    # Deal with vmin and vmax
    if (vmin is not None) ^ (vmax is not None):
        raise ValueError("'vmin' and 'vmax' must be defined together or both equal to None.")

    vmin = min(vmin, torch.min(cka_matrix).item()) if vmin is not None else vmin
    vmax = max(vmax, torch.max(cka_matrix).item()) if vmax is not None else vmax

    # Build the mask
    if show_half_heatmap:
        mask = torch.tril(torch.ones_like(cka_matrix, dtype=torch.bool), diagonal=-1).cpu().numpy()
    else:
        mask = None

    # Build the heatmap
    ax = sn.heatmap(cka_matrix.cpu(), vmin=vmin, vmax=vmax, annot=show_annotations, cmap=cmap, mask=mask)
    if invert_y_axis:
        ax.invert_yaxis()

    ax.set_xlabel(f"{second_name} layers", fontsize=12)
    ax.set_ylabel(f"{first_name} layers", fontsize=12)

    # Deal with tick labels
    if show_ticks_labels:
        if short_tick_labels_splits is None:
            ax.set_xticklabels(second_name)
            ax.set_yticklabels(first_name)
        else:
            ax.set_xticklabels(["-".join(module.split(".")[-short_tick_labels_splits:]) for module in second_layers])
            ax.set_yticklabels(["-".join(module.split(".")[-short_tick_labels_splits:]) for module in first_layers])

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

    # Put the title if passed
    if title is not None:
        ax.set_title(title, fontsize=14)
    else:
        title = f"{first_name} vs {second_name}"
        ax.set_title(title, fontsize=14)

    # Set the layout to tight if the corresponding parameter is True
    if use_tight_layout:
        plt.tight_layout()

    # Save the plot to the specified path if defined
    if save_path is not None:
        title = title.replace(" ", "_").replace("/", "-")
        path_rel = f"{save_path}/{title}.png"
        plt.savefig(path_rel, dpi=400, bbox_inches="tight")

    # Show the image if the user chooses to do so
    if show_img:
        plt.show()
