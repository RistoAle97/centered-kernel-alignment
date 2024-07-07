import torch
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader, TensorDataset

from cka import CKA


class FF(nn.Module):

    def __init__(self) -> None:
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
        super().__init__()
        self.first = FF()
        self.second = FF()
        self.third = FF()
        self.fourth = FF()
        self.fifth = FF()
        self.sixth = FF()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        out = self.first(src)
        out = self.second(out)
        out = self.third(out)
        out = self.fourth(out)
        out = self.fifth(out)
        out = self.sixth(out)
        return out


if __name__ == "__main__":
    torch.manual_seed(5)
    test_model = ManyFF()
    test_model_new = ManyFF()
    layers_to_observe = [
        "first.linear1", "second.linear1", "third.linear1", "fourth.linear1", "fifth.linear1", "sixth.linear1",
    ]
    cka = CKA(
        first_model=test_model,
        second_model=test_model,
        layers=layers_to_observe,
        first_name="ManyFF",
        second_name="ManyFF_new",
        device="cuda:0",
    )
    x_new = torch.randn(32, 128, 512, dtype=torch.float64)
    dataset = TensorDataset(x_new)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0, drop_last=True)
    cka_matrix = cka(dataloader)
    cka.plot_cka(
        cka_matrix=cka_matrix,
        title=f"Model comparison, bsz {dataloader.batch_size}",
        show_ticks_labels=True,
        short_tick_labels_splits=2,
        use_tight_layout=True,
    )
