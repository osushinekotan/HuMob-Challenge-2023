import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, feature_seqs, auxiliary_seqs, target_seqs):
        self.feature_seqs = feature_seqs
        self.auxiliary_seqs = auxiliary_seqs
        self.target_seqs = target_seqs

    def __len__(self):
        return len(self.feature_seqs)

    def __getitem__(self, index: int) -> dict[str : torch.Tensor]:
        feature_seqs = torch.Tensor(self.feature_seqs[index]).float()
        auxiliary_seqs = torch.Tensor(self.auxiliary_seqs[index]).float()
        target_seqs = torch.Tensor(self.target_seqs[index]).float()
        return {
            "feature_seqs": feature_seqs,
            "auxiliary_seqs": auxiliary_seqs,
            "target_seqs": target_seqs,
        }


class TestDataset(Dataset):
    def __init__(self, feature_seqs, auxiliary_seqs):
        self.feature_seqs = feature_seqs
        self.auxiliary_seqs = auxiliary_seqs

    def __len__(self):
        return len(self.feature_seqs)

    def __getitem__(self, index: int) -> dict[str : torch.Tensor]:
        feature_seqs = torch.Tensor(self.feature_seqs[index]).float()
        auxiliary_seqs = torch.Tensor(self.auxiliary_seqs[index]).float()
        return {
            "feature_seqs": feature_seqs,
            "auxiliary_seqs": auxiliary_seqs,
        }


class TrainDatasetV02(Dataset):
    def __init__(
        self,
        feature_seqs,
        auxiliary_seqs,
        target_seqs,
        central_node_feature_seqs,
        neighbor_node_feature_seqs,
    ):
        self.feature_seqs = feature_seqs
        self.auxiliary_seqs = auxiliary_seqs
        self.target_seqs = target_seqs

        self.central_node_feature_seqs = central_node_feature_seqs
        self.neighbor_node_feature_seqs = neighbor_node_feature_seqs

    def __len__(self):
        return len(self.feature_seqs)

    def __getitem__(self, index: int) -> dict[str : torch.Tensor]:
        feature_seqs = torch.Tensor(self.feature_seqs[index]).float()
        auxiliary_seqs = torch.Tensor(self.auxiliary_seqs[index]).float()
        target_seqs = torch.Tensor(self.target_seqs[index]).float()

        central_node_feature_seqs = torch.Tensor(self.central_node_feature_seqs[index]).float()
        neighbor_node_feature_seqs = torch.Tensor(self.neighbor_node_feature_seqs[index]).float()
        return {
            "feature_seqs": feature_seqs,
            "auxiliary_seqs": auxiliary_seqs,
            "target_seqs": target_seqs,
            "central_node_feature_seqs": central_node_feature_seqs,
            "neighbor_node_feature_seqs": neighbor_node_feature_seqs,
        }


class TestDatasetV02(Dataset):
    def __init__(
        self,
        feature_seqs,
        auxiliary_seqs,
        central_node_feature_seqs,
        neighbor_node_feature_seqs,
    ):
        self.feature_seqs = feature_seqs
        self.auxiliary_seqs = auxiliary_seqs
        self.central_node_feature_seqs = central_node_feature_seqs
        self.neighbor_node_feature_seqs = neighbor_node_feature_seqs

    def __len__(self):
        return len(self.feature_seqs)

    def __getitem__(self, index: int) -> dict[str : torch.Tensor]:
        feature_seqs = torch.Tensor(self.feature_seqs[index]).float()
        auxiliary_seqs = torch.Tensor(self.auxiliary_seqs[index]).float()
        central_node_feature_seqs = torch.Tensor(self.central_node_feature_seqs[index]).float()
        neighbor_node_feature_seqs = torch.Tensor(self.neighbor_node_feature_seqs[index]).float()
        return {
            "feature_seqs": feature_seqs,
            "auxiliary_seqs": auxiliary_seqs,
            "central_node_feature_seqs": central_node_feature_seqs,
            "neighbor_node_feature_seqs": neighbor_node_feature_seqs,
        }
