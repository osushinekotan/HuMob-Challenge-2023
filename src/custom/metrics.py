import torch
from torch.nn.utils.rnn import pad_sequence


def pad_sequences_with_torch(sequences, padding_value):
    sequences = [torch.tensor(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    return padded_sequences


class MaskedMSELoss:
    def __init__(self, padding_value=0):
        self.padding_value = padding_value

    def __call__(self, output, target):
        target_padded = pad_sequences_with_torch(target, padding_value=self.padding_value)
        output_padded = pad_sequences_with_torch(output, padding_value=self.padding_value)
        mask = target_padded.ne(self.padding_value)
        loss = (output_padded - target_padded) ** 2
        loss = loss * mask.float()
        return loss.sum() / mask.sum().float()
