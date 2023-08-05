import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


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


def to_device(batch, device):
    for k, v in batch.items():
        batch[k] = v.to(device)
    return batch


def train_fn(config, wandb_logger):
    model = config["/model"]
    dataloader = config["/dataloader/train"]
    criterion = config["/criterion"]
    optimizer = config["/optimizer"]
    scheduler = config["/scheduler"]

    # training settings
    device = config["/nn/device"]
    use_amp = config["/nn/fp16"]
    gradient_accumulation_steps = config["/nn/gradient_accumulation_steps"]
    clip_grad_norm = config["/nn/clip_grad_norm"]

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
        if config.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if config.batch_scheduler:
                scheduler.step()

        wandb_logger.log({"train_loss": loss, "lr": scheduler.get_lr()[0]})
        losses.append(float(loss))
        iteration_bar.set_description(f"loss: {np.mean(losses):.4f} lr: {scheduler.get_lr()[0]:.6f}")

    loss = np.mean(losses)
    return {"loss": loss, "step": step}


def valid_fn(config):
    model = config["/model"]
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

    outputs = np.concatenate(outputs)
    loss = np.mean(losses)
    return {"loss": loss, "outputs": outputs}
