import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from run_multimodal_v14 import ModerateDataTransform, MultimodalBIOT_V14, mixup_criterion, mixup_data


DEFAULT_TRAIN_SUBJECTS = [
    "VP008",
    "VP001",
    "VP012",
    "VP019",
    "VP015",
    "VP007",
    "VP025",
    "VP009",
    "VP004",
    "VP002",
    "VP016",
    "VP003",
    "VP005",
    "VP023",
    "VP024",
    "VP006",
]
DEFAULT_VAL_SUBJECTS = ["VP021", "VP011", "VP014", "VP018", "VP022"]
DEFAULT_TEST_SUBJECTS = ["VP017", "VP013", "VP010", "VP026", "VP020"]


class ShinSubjectSplitDataset(Dataset):
    def __init__(self, root_path, subjects, transform=None):
        root = Path(root_path)
        subject_set = set(subjects)
        self.paths = sorted(path for path in root.glob("*.pkl") if path.name.split("_")[0] in subject_set)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def normalize(x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + 1e-5)

    def __getitem__(self, idx):
        with self.paths[idx].open("rb") as f:
            data = pickle.load(f)
        x_eeg = torch.from_numpy(data["X_eeg"]).float()
        x_nirs = torch.from_numpy(data["X_nirs"]).float()
        y = torch.tensor(data["y"]).long()
        x_eeg = self.normalize(x_eeg)
        x_nirs = self.normalize(x_nirs)
        if self.transform:
            x_eeg = self.transform(x_eeg)
            x_nirs = self.transform(x_nirs)
        return x_eeg, x_nirs, y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_eeg, x_nirs, y in loader:
            logits = model(x_eeg.to(device), x_nirs.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            y_true.extend(y.numpy().tolist())
    labels = [0, 1, 2]
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def build_model(args, device):
    eeg_args = {
        "emb_size": args.emb_size,
        "heads": args.heads,
        "depth": args.depth,
        "n_channels": args.in_channels_eeg,
        "n_fft": args.token_size,
        "hop_length": args.hop_length,
    }
    model = MultimodalBIOT_V14(
        eeg_args=eeg_args,
        nirs_channels=args.in_channels_nirs,
        n_classes=args.n_classes,
    ).to(device)
    if args.checkpoint_path:
        print(f"Loading checkpoint: {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    for param in model.parameters():
        if param.dtype.is_floating_point:
            param.requires_grad = True
    return model


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_subjects = args.train_subjects or DEFAULT_TRAIN_SUBJECTS
    val_subjects = args.val_subjects or DEFAULT_VAL_SUBJECTS
    test_subjects = args.test_subjects or DEFAULT_TEST_SUBJECTS
    split = {"train": train_subjects, "val": val_subjects, "test": test_subjects}
    (save_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    train_transform = ModerateDataTransform(noise_std=args.noise_std, mask_ratio=args.mask_ratio)
    train_set = ShinSubjectSplitDataset(args.data_path, train_subjects, transform=train_transform)
    val_set = ShinSubjectSplitDataset(args.data_path, val_subjects)
    test_set = ShinSubjectSplitDataset(args.data_path, test_subjects)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Using device: {device}")
    print("Train subjects:", train_subjects)
    print("Val subjects:", val_subjects)
    print("Test subjects:", test_subjects)
    print(f"Dataset Size: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    model = build_model(args, device)
    eeg_params = [p for p in model.eeg_encoder.parameters() if p.requires_grad]
    eeg_param_ids = set(map(id, eeg_params))
    base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in eeg_param_ids]
    optimizer = AdamW(
        [
            {"params": eeg_params, "lr": args.lr * args.eeg_lr_scale},
            {"params": base_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_patience)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x_eeg, x_nirs, y in train_loader:
            x_eeg = x_eeg.to(device)
            x_nirs = x_nirs.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            mixed_x_eeg, mixed_x_nirs, y_a, y_b, lam = mixup_data(
                x_eeg,
                x_nirs,
                y,
                alpha=args.mixup_alpha,
                device=device,
            )
            logits = model(mixed_x_eeg, mixed_x_nirs)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["f1_macro"])
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "lr_eeg": optimizer.param_groups[0]["lr"],
            "lr_base": optimizer.param_groups[1]["lr"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"LR(E):{record['lr_eeg']:.1e} LR(B):{record['lr_base']:.1e} "
            f"Loss:{record['train_loss']:.4f} "
            f"ValAcc:{val_metrics['acc']:.4f} ValF1:{val_metrics['f1_macro']:.4f}"
        )
        print("Val confusion [rows=true, cols=pred]:", val_metrics["confusion"])

        if val_metrics["f1_macro"] > best_val_f1 + args.min_delta:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), save_dir / "best_v14_subject_split.pth")
        else:
            stale_epochs += 1

        if args.early_stop_patience and stale_epochs >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, best_val_f1={best_val_f1:.4f}")
            break

    model.load_state_dict(torch.load(save_dir / "best_v14_subject_split.pth", map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    final_record = {
        "best_epoch": best_epoch,
        "monitor_f1_macro": best_val_f1,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    (save_dir / "final_test_metrics.json").write_text(json.dumps(final_record, indent=2), encoding="utf-8")
    print(
        f"Final test | best_epoch={best_epoch} "
        f"acc={test_metrics['acc']:.4f} f1={test_metrics['f1_macro']:.4f}"
    )
    print("Final test confusion [rows=true, cols=pred]:", test_metrics["confusion"])
    print(f"Saved run to {save_dir}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/Shin2018/processed_aug_stride2/all")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--save_dir", default="runs/v14_subject_split_seed42")
    parser.add_argument("--train_subjects", nargs="*")
    parser.add_argument("--val_subjects", nargs="*")
    parser.add_argument("--test_subjects", nargs="*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in_channels_eeg", type=int, default=30)
    parser.add_argument("--in_channels_nirs", type=int, default=36)
    parser.add_argument("--n_classes", type=int, default=3)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eeg_lr_scale", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--mask_ratio", type=float, default=0.05)
    parser.add_argument("--token_size", type=int, default=200)
    parser.add_argument("--hop_length", type=int, default=100)
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
