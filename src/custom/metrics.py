import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class MSE:
    def __init__(self, squared=True):
        self.squared = squared

    def __call__(self, output, target):
        return mean_squared_error(target, output, squared=self.squared)


def pad_sequences_with_torch(sequences, padding_value):
    sequences = [torch.tensor(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    return padded_sequences


class MaskedMSEMetrics:
    def __init__(self, padding_value=0):
        self.padding_value = padding_value

    def __call__(self, output, target):
        target_padded = pad_sequences_with_torch(target, padding_value=self.padding_value)
        output_padded = pad_sequences_with_torch(output, padding_value=self.padding_value)
        mask = target_padded.ne(self.padding_value)
        loss = (output_padded - target_padded) ** 2
        loss = loss * mask.float()
        return loss.sum() / mask.sum().float()


class MaskedMSELoss(nn.Module):
    def __init__(self, padding_value=0.0):
        super(MaskedMSELoss, self).__init__()
        self.padding_value = padding_value

    def forward(self, output, target):
        mask = target != self.padding_value

        loss = (output - target) ** 2
        loss = loss * mask.float()

        return loss.sum() / mask.float().sum()
