import numpy as np
import torch
from pytorch_pfn_extras.config import Config
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class PadSequenceCollateFn:
    def __init__(self, is_train_mode=True, padding_value=-1, return_padding_mask=False):
        self.is_train_mode = is_train_mode
        self.padding_value = padding_value
        self.return_padding_mask = return_padding_mask

    def __call__(self, batch):
        feature_seqs = [item["feature_seqs"] for item in batch]
        auxiliary_seqs = [item["auxiliary_seqs"] for item in batch]
        feature_lengths = [len(seq) for seq in feature_seqs]
        auxiliary_lengths = [len(seq) for seq in auxiliary_seqs]

        feature_seqs_padded = pad_sequence(
            [(seq) for seq in feature_seqs],
            batch_first=True,
            padding_value=self.padding_value,
        )  # (sequence_len, feature_dim)
        auxiliary_seqs_padded = pad_sequence(
            [(seq) for seq in auxiliary_seqs],
            batch_first=True,
            padding_value=self.padding_value,
        )  # (sequence_len, feature_dim)

        if not self.is_train_mode:
            batch = {
                "feature_seqs": feature_seqs_padded,
                "auxiliary_seqs": auxiliary_seqs_padded,
                "feature_lengths": feature_lengths,
                "auxiliary_lengths": auxiliary_lengths,
            }
            if self.return_padding_mask:
                batch = {
                    **batch,
                    **{
                        "feature_padding_mask": (feature_seqs_padded[:, :, 0] == self.padding_value).bool(),
                        "auxiliary_padding_mask": (auxiliary_seqs_padded[:, :, 0] == self.padding_value).bool(),
                    },
                }
            return batch

        target_seqs = [item["target_seqs"] for item in batch]
        target_seqs_padded = pad_sequence(
            [(seq) for seq in target_seqs],
            batch_first=True,
            padding_value=self.padding_value,
        )  # (sequence_len, target_dim)

        batch = {
            "feature_seqs": feature_seqs_padded,
            "auxiliary_seqs": auxiliary_seqs_padded,
            "target_seqs": target_seqs_padded,
            "feature_lengths": feature_lengths,
            "auxiliary_lengths": auxiliary_lengths,
        }
        if self.return_padding_mask:
            batch = {
                **batch,
                **{
                    "feature_padding_mask": (feature_seqs_padded[:, :, 0] == self.padding_value).bool(),
                    "auxiliary_padding_mask": (auxiliary_seqs_padded[:, :, 0] == self.padding_value).bool(),
                },
            }
        return batch


def to_device(batch, device):
    for k, v in batch.items():
        if not k.endswith("lengths"):
            batch[k] = v.to(device)
    return batch


def train_fn(
    config: Config,
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    wandb_logger,
    total_step,
):
    # training settings
    device = config["/nn/device"]
    use_amp = config["/nn/fp16"]
    gradient_accumulation_steps = config["/nn/gradient_accumulation_steps"]
    clip_grad_norm = config["/nn/clip_grad_norm"]
    batch_scheduler = config["/nn/batch_scheduler"]

    model.to(device)
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    losses = []

    iteration_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch in iteration_bar:
        batch = to_device(batch, device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            batch_outputs = model(batch)
            loss = criterion(
                batch_outputs,
                batch["target_seqs"],
                target_len=batch["auxiliary_lengths"],
            )
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

        losses.append(float(loss))
        ave_loss = np.mean(losses)
        iteration_bar.set_description(f"step: {total_step}, loss: {ave_loss:.4f} lr: {scheduler.get_last_lr()[0]:.6f}")
        if wandb_logger is not None:
            wandb_logger.log(
                {
                    "train_ave_loss": ave_loss,
                    "train_loss": float(loss),
                    "lr": scheduler.get_last_lr()[0],
                    "train_step": total_step,
                }
            )

    loss = np.mean(losses)
    if not batch_scheduler:
        scheduler.step()

    return {"loss": loss, "step": total_step}


def valid_fn(config: Config, model, dataloader):
    dataloader = config["/nn/dataloader/valid"]
    criterion = config["/nn/criterion"]

    # training settings
    device = config["/nn/device"]
    gradient_accumulation_steps = config["/nn/gradient_accumulation_steps"]

    model.to(device)
    model.eval()
    outputs, losses = [], []

    iteration_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, batch in iteration_bar:
        batch = to_device(batch, device)

        with torch.no_grad():
            batch_outputs = model(batch)
            loss = criterion(
                batch_outputs,
                batch["target_seqs"],
                target_len=batch["auxiliary_lengths"],
            )
            loss = torch.div(loss, gradient_accumulation_steps)

        batch_outputs = batch_outputs.to("cpu").numpy()
        for a_batch_outputs, a_length in zip(batch_outputs, batch["auxiliary_lengths"]):
            outputs.append(a_batch_outputs[:a_length])

        losses.append(float(loss))
        iteration_bar.set_description(f"loss: {np.mean(losses):.4f}")

    outputs = np.concatenate(outputs)
    loss = np.mean(losses)
    return {"loss": loss, "outputs": outputs}


def inference_fn(config: Config, model):
    device = config["/nn/device"]
    dataloader = config["/nn/dataloader/test"]

    model.eval()
    model.to(device)
    iteration_bar = tqdm(dataloader, total=len(dataloader))
    outputs = []
    for batch in iteration_bar:
        batch = to_device(batch, device)

        with torch.no_grad():
            batch_outputs = model(batch)

        batch_outputs = batch_outputs.cpu().detach().numpy()
        for a_bach, a_length in zip(batch_outputs, batch["auxiliary_lengths"]):
            outputs.append(a_bach[:a_length])

    outputs = np.concatenate(outputs)
    return outputs
