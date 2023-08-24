import warnings

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Ignore specific message
warnings.filterwarnings("ignore", "Lazy modules are a new feature under heavy development.*")


class CustomLSTMModelV1(nn.Module):
    def __init__(
        self,
        input_size1,
        input_size2,
        hidden_size,
        num_layers,
        dropout,
        output_size,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size1,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder = nn.LSTM(
            input_size2,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

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
        x1, (hn_1, cn_1) = self.encoder(s1)

        # Use the final hidden and cell state of encoder as initial state for decoder
        x2, _ = self.decoder(s2, (hn_1, cn_1))
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
        """Oneshot sequence prediction model"""

        self.embedding_src = nn.Linear(input_size_src, d_model)
        self.embedding_tgt = nn.Linear(input_size_tgt, d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.out = nn.Linear(d_model, output_size)

    def forward(self, batch):
        x_src = self.embedding_src(batch["feature_seqs"])
        x_tgt = self.embedding_tgt(batch["auxiliary_seqs"])

        src_mask = batch["feature_padding_mask"]
        tgt_mask = batch["auxiliary_padding_mask"]

        encoder_output = self.encoder(x_src, src_key_padding_mask=src_mask)
        decoder_output = self.decoder(tgt=x_tgt, memory=encoder_output, tgt_key_padding_mask=tgt_mask)

        x = self.out(decoder_output)

        return x


# =======================================
# Attention seq2seq
# =======================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return nn.functional.softmax(attention, dim=1)


class DecoderWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.attention = Attention(hidden_dim)

        self.lstm = nn.LSTM(
            input_dim + hidden_dim * 2,  # Attentionの出力を含むため
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_out = nn.Linear(hidden_dim * 4, hidden_dim)  # 注意とLSTM出力の結合

    def forward(self, input, hidden, cell, encoder_outputs):
        attention_weights = self.attention(hidden[-1], encoder_outputs)
        attention_weights = attention_weights.unsqueeze(1)

        weighted = torch.bmm(attention_weights, encoder_outputs)
        lstm_input = torch.cat((input, weighted), dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        prediction = self.fc_out(torch.cat((output, weighted), dim=2))

        return prediction, hidden, cell


class CustomLSTMModelV1WithAttention(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, num_layers, dropout, output_size):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size1,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder = DecoderWithAttention(input_size2, hidden_size, num_layers, dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, batch):
        s1 = pack_padded_sequence(
            batch["feature_seqs"],
            batch["feature_lengths"],
            batch_first=True,
            enforce_sorted=False,
        )
        encoder_outputs, (hn_1, cn_1) = self.encoder(s1)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)

        auxiliary_seqs = batch["auxiliary_seqs"]

        outputs = torch.zeros(
            auxiliary_seqs.size(0),
            auxiliary_seqs.size(1),
            self.decoder.hidden_dim,
        ).to(auxiliary_seqs.device)

        for t in range(auxiliary_seqs.size(1)):
            output, hn_1, cn_1 = self.decoder(auxiliary_seqs[:, [t], :], hn_1, cn_1, encoder_outputs)
            outputs[:, t, :] = output.squeeze(1)
        x = self.out(outputs)

        return x
