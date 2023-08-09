import torch
from sklearn.metrics import mean_squared_error
from torch import nn


class MSE:
    def __init__(self, squared=True, higher_is_better=True):
        self.squared = squared
        self.higher_is_better = higher_is_better

    def __call__(self, output, target):
        score = mean_squared_error(target, output, squared=self.squared)
        score = -score if self.higher_is_better else score
        return score


class SeqMSELoss(nn.MSELoss):
    def forward(self, output, target, target_len: list[int]):
        output, target = self.flatten(output=output, target=target, target_len=target_len)
        return super(SeqMSELoss, self).forward(output, target)  # Call the forward method of the parent class

    @staticmethod
    def flatten(output, target, target_len: list[int]):
        # Using list comprehensions
        outputs = [i_output[:i_length] for i_output, i_length in zip(output, target_len)]
        targets = [i_target[:i_length] for i_target, i_length in zip(target, target_len)]

        return torch.cat(outputs, dim=0), torch.cat(targets, dim=0)
