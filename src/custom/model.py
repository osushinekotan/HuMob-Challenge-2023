import warnings

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Ignore specific message
warnings.filterwarnings("ignore", "Lazy modules are a new feature under heavy development.*")


class CustomLSTMModelV1(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(input_size1, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(input_size2, hidden_size, batch_first=True)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, batch):
        # to variable length
        s1 = pack_padded_sequence(
            batch["feature_seqs"],
            batch["feature_lengths"],
            batch_first=True,
            enforce_sorted=False,
        )
        s2 = pack_padded_sequence(
            batch["auxiliary_seqs"],
            batch["auxiliary_lengths"],
            batch_first=True,
            enforce_sorted=False,
        )
        x1, (hn_1, cn_1) = self.lstm1(s1)

        # Use the final hidden and cell state of lstm1 as initial state for lstm2
        x2, _ = self.lstm2(s2, (hn_1, cn_1))
        x, _ = pad_packed_sequence(x2, batch_first=True)  # to fixible length
        x = self.out(x)

        return x
