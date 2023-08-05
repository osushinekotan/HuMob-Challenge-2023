from torch.nn.utils.rnn import pad_sequence


class PadSequenceCollateFn:
    def __init__(self, is_train_mode=True):
        self.is_train_mode = is_train_mode

    def __call__(self, batch):
        feature_seqs = [item["feature_seqs"] for item in batch]
        lengths = [len(seq) for seq in feature_seqs]
        feature_seqs_padded = pad_sequence(
            [(seq) for seq in feature_seqs], batch_first=True
        )  # (sequence_len, feature_dim)

        if not self.is_train_mode:
            return {
                "feature_seqs": feature_seqs_padded,
                "lengths": lengths,
            }

        target_seqs = [item["target_seqs"] for item in batch]
        target_seqs_padded = pad_sequence(
            [(seq) for seq in target_seqs], batch_first=True
        )  # (sequence_len, target_dim)
        return {
            "feature_seqs": feature_seqs_padded,
            "target_seqs": target_seqs_padded,
            "lengths": lengths,
        }
