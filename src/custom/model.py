import warnings

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Ignore specific message
warnings.filterwarnings("ignore", "Lazy modules are a new feature under heavy development.*")


class CustomLSTMModelV1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, batch):
        # to variable length
        x = pack_padded_sequence(
            batch["feature_seqs"],
            batch["lengths"],
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # to fixible length
        out = self.fc(out)
        return out
