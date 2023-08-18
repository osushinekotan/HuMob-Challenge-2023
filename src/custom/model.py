import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Ignore specific message
warnings.filterwarnings("ignore", "Lazy modules are a new feature under heavy development.*")


class CustomLSTMModelV1(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(input_size1, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size2, hidden_size, batch_first=True, bidirectional=True)

        self.out = nn.Linear(hidden_size * 2, output_size)

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


class CustomTransformerModelV1(nn.Module):
    def __init__(
        self,
        input_size_src,
        input_size_tgt,
        d_model,
        output_size,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        super().__init__()
        self.embedding_src = nn.Linear(input_size_src, d_model)
        self.embedding_tgt = nn.Linear(input_size_tgt, d_model)

        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, output_size)

    def forward(self, batch):
        x_src = self.embedding_src(batch["feature_seqs"])
        x_tgt = self.embedding_tgt(batch["auxiliary_seqs"])

        src_mask = batch["feature_padding_mask"]
        tgt_mask = batch["auxiliary_padding_mask"]

        x = self.transformer(
            src=x_src,
            tgt=x_tgt,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
        )
        x = self.out(x)

        return x


# GraphSAGE with RNN-like update mechanism
class GraphGRUCell(nn.Module):
    def __init__(self, in_features, hidden_features, embed_features):
        super().__init__()

        self.node_feature_embedding = nn.Sequential(
            nn.Linear(in_features, embed_features),
            nn.ReLU(),
        )

        self.combine_linear = nn.Linear(embed_features + hidden_features, hidden_features)
        self.update_gate = nn.Linear(embed_features + hidden_features, hidden_features)
        self.reset_gate = nn.Linear(embed_features + hidden_features, hidden_features)

        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初期値として0.5を使用

    def forward(self, central_node_features, neighbor_node_features, mask, hidden_state):
        # aggregated_neighbors shape: (batch_size, in_features)
        aggregated_neighbors = (neighbor_node_features * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1)

        central_node_features = self.node_feature_embedding(central_node_features)
        aggregated_neighbors = self.node_feature_embedding(aggregated_neighbors)

        combined_features = self.alpha * central_node_features + (1 - self.alpha) * aggregated_neighbors
        combined = torch.cat([combined_features, hidden_state], dim=1)  # concat along feature dimension

        z = torch.sigmoid(self.update_gate(combined))
        r = torch.sigmoid(self.reset_gate(combined))

        combined_reset = torch.cat([central_node_features, r * hidden_state], dim=1)
        h_tilda = F.relu(self.combine_linear(combined_reset))
        new_hidden = (1 - z) * hidden_state + z * h_tilda

        return new_hidden  # shape: (batch_size, hidden_features)


class DynamicGraphLSTM(nn.Module):
    def __init__(
        self,
        in_features_sage,
        hidden_features_sage,
        embed_features_sage,
        input_size1_lstm,
        input_size2_lstm,
        hidden_size_lstm,
        output_size,
    ):
        super().__init__()

        self.graphsage_cell = GraphGRUCell(in_features_sage, hidden_features_sage, embed_features=embed_features_sage)
        self.hidden_features_sage = hidden_features_sage
        self.hidden_size_lstm = hidden_size_lstm

        self.lstm_cell_1 = nn.LSTMCell(hidden_features_sage + input_size1_lstm, hidden_size_lstm)
        self.lstm2 = nn.LSTM(input_size2_lstm, hidden_size_lstm, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_size_lstm * 2, output_size)

    def forward(
        self,
        central_node_features,
        neighbor_node_features,
        mask,
        feature_seqs,
        auxiliary_seqs,
    ):
        batch_size, sequence_len, _ = central_node_features.shape
        hidden_features_sage = self.hidden_features_sage
        hidden_features_lstm = self.hidden_size_lstm

        h_sage = torch.zeros(batch_size, hidden_features_sage, device=central_node_features.device)
        h_lstm1 = torch.zeros(batch_size, hidden_features_lstm, device=central_node_features.device)
        c_lstm1 = torch.zeros(batch_size, hidden_features_lstm, device=central_node_features.device)

        for i in range(sequence_len):
            h_sage = self.graphsage_cell(
                central_node_features[:, i, :],
                neighbor_node_features[:, i, :, :],
                mask[:, i, :],
                h_sage,
            )  # shape: (batch_size, hidden_features_sage)

            combined_input = torch.cat(
                [h_sage, feature_seqs[:, i, :]], dim=1
            )  # shape: (batch_size, hidden_features_sage + input_size1_lstm)
            h_lstm1, c_lstm1 = self.lstm_cell_1(combined_input, (h_lstm1, c_lstm1))

        h_lstm1 = h_lstm1.repeat(2, 1, 1)  # to input bidirectional
        c_lstm1 = c_lstm1.repeat(2, 1, 1)
        lstm2_outputs, _ = self.lstm2(
            auxiliary_seqs, (h_lstm1, c_lstm1)
        )  # shape: (batch_size, sequence_len, hidden_size_lstm * 2)
        outputs = self.out(lstm2_outputs)  # shape: (batch_size, sequence_len, output_size)

        return outputs
