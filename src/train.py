"""
Training loop.

Run with:
    python -m src.train

Checkpoints go to outputs/best_model.pt (best validation Dice so far) and
outputs/last_model.pt (most recent epoch, for resuming). TensorBoard logs
go to outputs/tensorboard/.
"""

from __future__ import annotations

import time

import torch
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from torch.utils.tensorboard import SummaryWriter

from .config import CONFIG, Config
from .dataset import build_datasets
from .model import build_model


def run_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    config: Config,
    post_pred: Compose,
    post_label: Compose,
    dice_metric: DiceMetric,
) -> float:
    model.eval()
    dice_metric.reset()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(config.device)
            labels = batch["label"].to(config.device)

            logits = sliding_window_inference(
                inputs=images,
                roi_size=config.patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

            preds = [post_pred(p) for p in decollate_batch(logits)]
            targets = [post_label(l) for l in decollate_batch(labels)]
            dice_metric(y_pred=preds, y=targets)

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return mean_dice


def train(config: Config = CONFIG) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "tensorboard").mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_datasets(config)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device.type == "cuda"),
    )
    # batch_size=1 for validation: cases can have different full-volume
    # sizes after CropForegroundd, so they can't always be batched together.
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config.num_workers)

    model = build_model(config)
    num_classes = config.label_scheme.num_classes

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    # Background is excluded from the reported Dice score: with it included,
    # a model that just predicts "everything is background" scores
    # deceptively high on mostly-background volumes.
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = Compose([AsDiscrete(to_onehot=num_classes)])

    writer = SummaryWriter(log_dir=str(config.output_dir / "tensorboard"))

    best_dice = -1.0
    epochs_since_improvement = 0

    for epoch in range(config.max_epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(config.device)
            labels = batch["label"].to(config.device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / max(1, len(train_loader))
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        print(f"epoch {epoch + 1}/{config.max_epochs}  loss={avg_loss:.4f}  "
              f"time={time.time() - epoch_start:.1f}s")

        torch.save(model.state_dict(), config.output_dir / "last_model.pt")

        if (epoch + 1) % config.val_interval == 0:
            mean_dice = run_validation(model, val_loader, config, post_pred, post_label, dice_metric)
            writer.add_scalar("val/dice", mean_dice, epoch)
            print(f"  validation dice (mean over foreground classes): {mean_dice:.4f}")

            if mean_dice > best_dice:
                best_dice = mean_dice
                epochs_since_improvement = 0
                torch.save(model.state_dict(), config.output_dir / "best_model.pt")
                print(f"  new best model saved (dice={best_dice:.4f})")
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= config.early_stop_patience:
                print(f"no improvement in {config.early_stop_patience} validation checks, stopping early")
                break

    writer.close()
    print(f"training done. best validation dice: {best_dice:.4f}")


if __name__ == "__main__":
    train()
