import torch
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader, TensorDataset

from cka import CKA


class FF(nn.Module):
    def __init__(self) -> None:
        """
        Simple feed-forward network like the ones used in the transformer architecture.
        """
        super().__init__()
        self.linear1 = nn.Linear(512, 2048, dtype=torch.float64)
        self.dropout = nn.Dropout(0.0)
        self.linear2 = nn.Linear(2048, 512, dtype=torch.float64)
        self.activation = F.relu

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear1(src))
        out = self.dropout(out)
        out = self.linear2(out)
        return out


class ManyFF(nn.Module):
    def __init__(self) -> None:
        """
        Many feed-forward networks one after the other.
        """
        super().__init__()
        self.first_ff = FF()
        self.second_ff = FF()
        self.third_ff = FF()
        self.fourth_ff = FF()
        self.fifth_ff = FF()
        self.sixth_ff = FF()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        out = self.first_ff(src)
        out = self.second_ff(out)
        out = self.third_ff(out)
        out = self.fourth_ff(out)
        out = self.fifth_ff(out)
        out = self.sixth_ff(out)
        return out


if __name__ == "__main__":
    # Set a seed for reproducibility
    torch.manual_seed(5)

    # Build the models
    first_many_ff = ManyFF()
    second_many_ff = ManyFF()

    # Define the layers that will be used during the computation
    layers_to_observe = [
        "first_ff.linear1",
        "second_ff.linear1",
        "third_ff.linear1",
        "fourth_ff.linear1",
        "fifth_ff.linear1",
        "sixth_ff.linear1",
    ]

    # Define the shared parameters between the two CKA objects
    shared_parameters = {
        "layers": layers_to_observe,
        "first_name": "ManyFF_0",
        "device": "cuda:0",
    }

    # Build the CKA objects, one for confronting a model with itself and the other for comparing two different models
    cka_same_model = CKA(
        first_model=first_many_ff,
        second_model=first_many_ff,
        **shared_parameters,
    )
    cka_different_models = CKA(
        first_model=first_many_ff,
        second_model=second_many_ff,
        second_name="ManyFF_1",
        **shared_parameters,
    )

    # Build the dataset from a random three-dimensional tensor and define the dataloader
    dataset = TensorDataset(torch.randn(128, 256, 512, dtype=torch.float64))
    dataloader = DataLoader(dataset, batch_size=8, num_workers=0, drop_last=True)

    # Compute the CKA values for both scenarios
    cka_matrix_same = cka_same_model(dataloader)
    cka_matrix_different = cka_different_models(dataloader)

    # Plot the CKA values
    plot_parameters = {
        "show_tick_labels": True,
        "short_tick_labels_splits": 2,
        "use_tight_layout": True,
        "show_half_heatmap": True,
    }
    cka_same_model.plot_cka(
        cka_matrix=cka_matrix_same,
        title=f"Model {cka_same_model.first_model_infos["name"]} compared with itself",
        **plot_parameters,
    )
    cka_different_models.plot_cka(cka_matrix=cka_matrix_different, **plot_parameters)
