"""Train student MLP with mixed distillation loss."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.student_mlp import StudentActionMLP


LOG = logging.getLogger(__name__)


class DistillDataset(Dataset):
    def __init__(self, npz_path: Path) -> None:
        payload = np.load(npz_path, allow_pickle=True)
        self.action_vectors = payload["action_vectors"].astype(np.float32)
        self.action_mask = payload["action_mask"].astype(bool)
        self.teacher_logits = payload["teacher_logits"].astype(np.float32)
        self.target_idx = payload["target_idx"].astype(np.int64)

    def __len__(self) -> int:
        return int(self.action_vectors.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "action_vectors": torch.tensor(self.action_vectors[idx], dtype=torch.float32),
            "action_mask": torch.tensor(self.action_mask[idx], dtype=torch.bool),
            "teacher_logits": torch.tensor(self.teacher_logits[idx], dtype=torch.float32),
            "target_idx": torch.tensor(self.target_idx[idx], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-layer-norm", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping on val loss")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs)")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum val loss improvement to reset patience")
    return parser.parse_args()


def _split_indices(n: int, val_split: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n)
    rng.shuffle(indices)
    val_count = int(n * float(val_split))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    return train_idx, val_idx


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.clone()
    masked[~mask] = -1e9
    return masked


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    dataset_path = Path(args.dataset)
    dataset = DistillDataset(dataset_path)
    total = len(dataset)
    rng = np.random.default_rng(int(args.seed))
    train_idx, val_idx = _split_indices(total, float(args.val_split), rng)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False)

    input_dim = int(dataset.action_vectors.shape[-1])
    device = torch.device(args.device)
    model = StudentActionMLP(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        use_layer_norm=bool(args.use_layer_norm),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    temperature = float(args.temperature)
    kl_weight = float(args.kl_weight)
    ce_loss_fn = nn.CrossEntropyLoss()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path("reports") / "distill" / f"student_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_path = run_dir / "student_mlp_best.pt"
    final_path = run_dir / "student_mlp_final.pt"

    def _run_epoch(loader: DataLoader, train: bool) -> Dict[str, float]:
        if train:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        total_ce = 0.0
        total_kl = 0.0
        total_acc = 0.0
        batches = 0
        for batch in loader:
            action_vectors = batch["action_vectors"].to(device)
            action_mask = batch["action_mask"].to(device)
            teacher_logits = batch["teacher_logits"].to(device)
            target_idx = batch["target_idx"].to(device)

            logits = model(action_vectors)
            logits_masked = _masked_logits(logits, action_mask)
            ce = ce_loss_fn(logits_masked, target_idx)

            teacher_masked = _masked_logits(teacher_logits, action_mask)
            teacher_probs = torch.softmax(teacher_masked / temperature, dim=-1)
            student_log_probs = torch.log_softmax(logits_masked / temperature, dim=-1)
            kl = (teacher_probs * (torch.log(teacher_probs + 1e-9) - student_log_probs)).sum(dim=-1).mean()
            loss = ce + kl_weight * (temperature ** 2) * kl

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits_masked, dim=-1)
                acc = (pred == target_idx).float().mean()

            total_loss += float(loss.item())
            total_ce += float(ce.item())
            total_kl += float(kl.item())
            total_acc += float(acc.item())
            batches += 1
        if batches == 0:
            return {"loss": 0.0, "ce": 0.0, "kl": 0.0, "acc": 0.0}
        return {
            "loss": total_loss / batches,
            "ce": total_ce / batches,
            "kl": total_kl / batches,
            "acc": total_acc / batches,
        }

    for epoch in range(1, int(args.epochs) + 1):
        train_stats = _run_epoch(train_loader, train=True)
        val_stats = _run_epoch(val_loader, train=False)
        LOG.info(
            "Epoch %d | train loss=%.4f acc=%.3f | val loss=%.4f acc=%.3f",
            epoch,
            train_stats["loss"],
            train_stats["acc"],
            val_stats["loss"],
            val_stats["acc"],
        )
        improved = val_stats["loss"] < (best_val_loss - float(args.min_delta))
        if improved:
            best_val_loss = float(val_stats["loss"])
            epochs_no_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "input_dim": input_dim,
                        "hidden_dim": int(args.hidden_dim),
                        "num_layers": int(args.num_layers),
                        "dropout": float(args.dropout),
                        "use_layer_norm": bool(args.use_layer_norm),
                    },
                },
                best_path,
            )
        else:
            epochs_no_improve += 1
            if args.early_stop and epochs_no_improve >= int(args.patience):
                LOG.info(
                    "Early stopping triggered (patience=%d, min_delta=%.6f).",
                    int(args.patience),
                    float(args.min_delta),
                )
                break

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "input_dim": input_dim,
                "hidden_dim": int(args.hidden_dim),
                "num_layers": int(args.num_layers),
                "dropout": float(args.dropout),
                "use_layer_norm": bool(args.use_layer_norm),
            },
        },
        final_path,
    )

    meta = {
        "dataset": str(dataset_path),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "temperature": float(args.temperature),
        "kl_weight": float(args.kl_weight),
        "val_split": float(args.val_split),
        "best_val_loss": float(best_val_loss),
        "best_model_path": str(best_path),
        "final_model_path": str(final_path),
    }
    (run_dir / "distill_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="ascii")
    LOG.info("Saved student model to %s", final_path)


if __name__ == "__main__":
    main()
