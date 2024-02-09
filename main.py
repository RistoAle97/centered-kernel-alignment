import torch
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader, TensorDataset

from cka import CKA, cka_base


class FF(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(512, 2048, dtype=torch.float64)
        self.dropout = nn.Dropout(0.0)
        self.linear2 = nn.Linear(2048, 512, dtype=torch.float64)
        self.activation = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        return out


class ManyFF(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.first = FF()
        self.second = FF()
        self.third = FF()
        self.fourth = FF()
        self.fifth = FF()
        self.sixth = FF()

    def forward(self, x) -> torch.Tensor:
        out = self.first(x)
        out = self.second(out)
        out = self.third(out)
        out = self.fourth(out)
        out = self.fifth(out)
        out = self.sixth(out)
        return out


if __name__ == "__main__":
    test_model = ManyFF()
    test_model_new = ManyFF()
    layers_to_observe = [
        "first.linear1", "second.linear1", "third.linear1", "fourth.linear1", "fifth.linear1", "sixth.linear1",
        "first.linear2", "second.linear2", "third.linear2", "fourth.linear2", "fifth.linear2", "sixth.linear2"
    ]
    cka = CKA(test_model, test_model_new, layers_to_observe, first_name="test_0", second_name="test_1", device="cuda:0")
    x = torch.randn(128, 512, dtype=torch.float64)
    x_new = torch.randn(10, 128, 512, dtype=torch.float64)
    y = torch.randn(128, 512, dtype=torch.float64)
    dataset = TensorDataset(x_new)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
    cka_matrix = cka(dataloader, False, rbf_threshold=0.9)
    cka_class = cka.compute_cka(x, y, False)
    cka_hsic = cka_base(x, y, "rbf", method="hsic")
    cka_norm = cka_base(x, y, "rbf", method="fro_norm")
    cka.plot_results(
        cka_matrix, title="Niko test", show_ticks_labels=True, short_tick_labels_splits=1, use_tight_layout=True
    )
