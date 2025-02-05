import torch
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader, TensorDataset

from ckatorch import CKA


class FF(nn.Module):
    def __init__(self) -> None:
        """Simple feed-forward network like the ones used in the transformer architecture."""
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
        """Many feed-forward networks one after the other."""
        super().__init__()
        self.model = nn.Sequential(FF(), FF(), FF(), FF(), FF(), FF())

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.model(src)


if __name__ == "__main__":
    # Set a seed for reproducibility
    torch.manual_seed(5)

    # Build the models
    first_many_ff = ManyFF()
    second_many_ff = ManyFF()

    # Define the layers that will be used during the computation
    layers_to_observe = [
        "model.0.linear1",
        "model.1.linear1",
        "model.2.linear1",
        "model.3.linear1",
        "model.4.linear1",
        "model.5.linear1",
    ]

    # Define the shared parameters between the two CKA objects
    shared_parameters = {
        "layers": layers_to_observe,
        "first_name": "ManyFF_0",
        "device": torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
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
    dataloader = DataLoader(dataset, batch_size=8, num_workers=8, shuffle=True)

    # Compute the CKA values for both scenarios
    cka_matrix_same = cka_same_model(dataloader)
    cka_matrix_different = cka_different_models(dataloader)

    # Plot the CKA values
    plot_parameters = {
        "show_ticks_labels": True,
        "short_tick_labels_splits": 2,
        "use_tight_layout": True,
        "show_half_heatmap": True,
    }
    cka_same_model.plot_cka(
        cka_matrix=cka_matrix_same,
        title=f"Model {cka_same_model.first_model_info.name} compared with itself",
        **plot_parameters,
    )
    cka_different_models.plot_cka(cka_matrix=cka_matrix_different, **plot_parameters)
