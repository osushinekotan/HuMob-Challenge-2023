import warnings

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Ignore specific message
warnings.filterwarnings("ignore", "Lazy modules are a new feature under heavy development.*")


# =======================================
# LSTM
# =======================================
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


# =======================================
# Transformer
# =======================================
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
        dropout=0.1,
    ):
        super().__init__()
        """Oneshot sequence prediction model"""

        self.embedding_src = nn.Sequential(
            nn.Linear(input_size_src, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.embedding_tgt = nn.Sequential(
            nn.Linear(input_size_tgt, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

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
# Transformer + LSTM
# =======================================
class CustomTransformerLSTMV1(nn.Module):
    def __init__(
        self,
        input_size_src,
        input_size_tgt,
        d_model,
        output_size,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1,
        hidden_size_lstm=512,
        num_layers_lstm=1,
        dropout_lstm=0,
    ):
        super().__init__()
        """Oneshot sequence prediction model"""

        self.embedding_src = nn.Sequential(
            nn.Linear(input_size_src, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.embedding_tgt = nn.Sequential(
            nn.Linear(input_size_tgt, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.lstm_encoder = nn.LSTM(
            d_model,
            hidden_size_lstm,
            num_layers=num_layers_lstm,
            dropout=dropout_lstm,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_decoder = nn.LSTM(
            d_model,
            hidden_size_lstm,
            num_layers=num_layers_lstm,
            dropout=dropout_lstm,
            batch_first=True,
            bidirectional=True,
        )

        self.out = nn.Linear(hidden_size_lstm * 2, output_size)

    def forward(self, batch):
        x_src = self.embedding_src(batch["feature_seqs"])
        x_tgt = self.embedding_tgt(batch["auxiliary_seqs"])

        src_mask = batch["feature_padding_mask"]
        tgt_mask = batch["auxiliary_padding_mask"]

        encoder_output = self.encoder(x_src, src_key_padding_mask=src_mask)
        decoder_output = self.decoder(tgt=x_tgt, memory=encoder_output, tgt_key_padding_mask=tgt_mask)

        _, (hn_1, cn_1) = self.lstm_encoder(encoder_output)
        x, _ = self.lstm_decoder(decoder_output, (hn_1, cn_1))

        x = self.out(x)
        return x


# =======================================
# Transformer + １DCNN
# =======================================
class CustomTransformer1DCNNV1(nn.Module):
    def __init__(
        self,
        input_size_src,
        input_size_tgt,
        d_model,
        output_size,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1,
        num_channels_cnn=[256, 512],  # 1D CNNのチャンネル数
        kernel_size_cnn=3,  # 1D CNNのカーネルサイズ
    ):
        super().__init__()
        """Oneshot sequence prediction model"""

        self.embedding_src = nn.Sequential(
            nn.Linear(input_size_src, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.embedding_tgt = nn.Sequential(
            nn.Linear(input_size_tgt, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # 1D CNN layers
        cnn_layers = []
        input_channels = d_model
        for output_channels in num_channels_cnn:
            cnn_layers.extend(
                [
                    nn.Conv1d(input_channels, output_channels, kernel_size_cnn, padding=kernel_size_cnn // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_channels = output_channels
        self.cnn = nn.Sequential(*cnn_layers)

        self.out = nn.Linear(input_channels, output_size)

    def forward(self, batch):
        x_src = self.embedding_src(batch["feature_seqs"])
        x_tgt = self.embedding_tgt(batch["auxiliary_seqs"])

        src_mask = batch["feature_padding_mask"]
        tgt_mask = batch["auxiliary_padding_mask"]

        encoder_output = self.encoder(x_src, src_key_padding_mask=src_mask)
        decoder_output = self.decoder(tgt=x_tgt, memory=encoder_output, tgt_key_padding_mask=tgt_mask)

        # 1D CNNによる処理
        # 注意：Conv1dは(N, C, L)の形式を期待しているので、次元を入れ替える必要があります
        cnn_out = self.cnn(decoder_output.permute(0, 2, 1))
        cnn_out = cnn_out.permute(0, 2, 1)

        x = self.out(cnn_out)

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


# =======================================
# LSTM + Transformer
# =======================================
class CustomLSTMTransformerV1(nn.Module):
    def __init__(
        self,
        input_size1,
        input_size2,
        hidden_size,
        dropout_lstm,
        num_layers_lstm,
        d_model,
        output_size,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout_transfomer=0.1,
    ):
        super().__init__()
        """Oneshot sequence prediction model"""

        self.lstm_encoder = nn.LSTM(
            input_size1,
            hidden_size,
            num_layers=num_layers_lstm,
            dropout=dropout_lstm,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_decoder = nn.LSTM(
            input_size2,
            hidden_size,
            num_layers=num_layers_lstm,
            dropout=dropout_lstm,
            batch_first=True,
            bidirectional=True,
        )

        self.embedding_src = nn.Sequential(
            nn.Linear(hidden_size * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout_transfomer),
        )
        self.embedding_tgt = nn.Sequential(
            nn.Linear(hidden_size * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout_transfomer),
        )

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.out = nn.Linear(d_model, output_size)

    def forward(self, batch):
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
        x1, (hn_1, cn_1) = self.lstm_encoder(s1)
        x2, _ = self.lstm_decoder(s2, (hn_1, cn_1))

        x1, _ = pad_packed_sequence(x1, batch_first=True)  # to fixible length
        x2, _ = pad_packed_sequence(x2, batch_first=True)

        x1 = self.embedding_src(x1)
        x2 = self.embedding_tgt(x2)

        encoder_output = self.encoder(x1)
        decoder_output = self.decoder(tgt=x2, memory=encoder_output)

        x = self.out(decoder_output)
        return x
