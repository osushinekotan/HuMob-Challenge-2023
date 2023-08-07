import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class PadSequenceCollateFn:
    def __init__(self, is_train_mode=True):
        self.is_train_mode = is_train_mode

    def __call__(self, batch):
        feature_seqs = [item["feature_seqs"] for item in batch]
        auxiliary_seqs = [item["auxiliary_seqs"] for item in batch]
        feature_lengths = [len(seq) for seq in feature_seqs]
        auxiliary_lengths = [len(seq) for seq in auxiliary_seqs]

        feature_seqs_padded = pad_sequence(
            [(seq) for seq in feature_seqs], batch_first=True
        )  # (sequence_len, feature_dim)
        auxiliary_seqs_padded = pad_sequence(
            [(seq) for seq in auxiliary_seqs], batch_first=True
        )  # (sequence_len, feature_dim)

        if not self.is_train_mode:
            return {
                "feature_seqs": feature_seqs_padded,
                "auxiliary_seqs": auxiliary_seqs_padded,
                "feature_lengths": feature_lengths,
                "auxiliary_lengths": auxiliary_lengths,
            }

        target_seqs = [item["target_seqs"] for item in batch]
        target_seqs_padded = pad_sequence(
            [(seq) for seq in target_seqs], batch_first=True
        )  # (sequence_len, target_dim)
        return {
            "feature_seqs": feature_seqs_padded,
            "auxiliary_seqs": auxiliary_seqs_padded,
            "target_seqs": target_seqs_padded,
            "feature_lengths": feature_lengths,
            "auxiliary_lengths": auxiliary_lengths,
        }


def to_device(batch, device):
    for k, v in batch.items():
        if not k.endswith("lengths"):
            batch[k] = v.to(device)
    return batch


def train_fn(config, model, wandb_logger, total_step):
    dataloader = config["/dataloader/train"]
    criterion = config["/criterion"]
    optimizer = config["/optimizer"]
    scheduler = config["/scheduler"]

    # training settings
    device = config["/nn/device"]
    use_amp = config["/nn/fp16"]
    gradient_accumulation_steps = config["/nn/gradient_accumulation_steps"]
    clip_grad_norm = config["/nn/clip_grad_norm"]
    batch_scheduler = config["/nn/batch_scheduler"]

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    losses = []

    iteration_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch in iteration_bar:
        batch = to_device(batch, device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            batch_outputs = model(batch)
            loss = criterion(batch_outputs, batch)
            loss = torch.div(loss, gradient_accumulation_steps)

        scaler.scale(loss).backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_step += 1

            if batch_scheduler:
                scheduler.step()

        if wandb_logger is not None:
            wandb_logger.log({"train_loss": loss, "lr": scheduler.get_lr()[0], "train_step": total_step})

        losses.append(float(loss))
        iteration_bar.set_description(f"loss: {np.mean(losses):.4f} lr: {scheduler.get_lr()[0]:.6f}")

    loss = np.mean(losses)
    if not batch_scheduler:
        scheduler.step()

    return {"loss": loss, "step": total_step}


def valid_fn(config, model):
    dataloader = config["/dataloader/valid"]
    criterion = config["/criterion"]

    # training settings
    device = config["/nn/device"]
    gradient_accumulation_steps = config["/nn/gradient_accumulation_steps"]

    model.eval()
    outputs, losses = [], []

    iteration_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, batch in iteration_bar:
        batch = to_device(batch, device)

        with torch.no_grad():
            batch_outputs = model(batch)
            loss = criterion(batch_outputs, batch)
            loss = torch.div(loss, gradient_accumulation_steps)

        batch_outputs = batch_outputs.to("cpu").numpy()
        outputs.append(batch_outputs)
        losses.append(float(loss))

        iteration_bar.set_description(f"loss: {np.mean(losses):.4f}")

    targets = np.concatenate([batch["targets"] for batch in dataloader])  # to store targets
    outputs = np.concatenate(outputs)
    loss = np.mean(losses)
    return {"loss": loss, "outputs": outputs, "targets": targets}
