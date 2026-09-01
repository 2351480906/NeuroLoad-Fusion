import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .data import ModerateDataTransform, ShinMultimodalDataset, mixup_criterion, mixup_data, seed_worker
from .modeling import MultimodalBIOT_V14


def load_partial_checkpoint(model, checkpoint_path, device):
    print(f"Loading partial V13 weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    loaded_state = {}
    skipped = []

    for key, value in checkpoint.items():
        if key in model_state and model_state[key].shape == value.shape:
            loaded_state[key] = value
        else:
            skipped.append(key)

    model_state.update(loaded_state)
    model.load_state_dict(model_state)

    skipped_prefixes = {}
    for key in skipped:
        prefix = key.split(".")[0]
        skipped_prefixes[prefix] = skipped_prefixes.get(prefix, 0) + 1

    print(f"  Loaded {len(loaded_state)}/{len(model_state)} tensors.")
    if skipped_prefixes:
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(skipped_prefixes.items()))
        print(f"  Skipped incompatible tensors by prefix: {summary}")


def reset_module_parameters(module):
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()


def set_module_trainable(module, trainable):
    for param in module.parameters():
        if param.dtype.is_floating_point:
            param.requires_grad = trainable


def build_optimizer(model, args):
    return AdamW([
        {"params": model.eeg_encoder.parameters(), "lr": args.lr_eeg, "name": "EEG"},
        {"params": model.eeg_adapter.parameters(), "lr": args.lr_adapter, "name": "Adapter"},
        {"params": model.nirs_preblock.parameters(), "lr": args.lr_nirs, "name": "PreBlock"},
        {
            "params": list(model.nirs_encoder.parameters()) + list(model.nirs_hb_encoder.parameters())
            + list(model.nirs_hbo_encoder.parameters()) + list(model.nirs_hbr_encoder.parameters()),
            "lr": args.lr_nirs_encoder,
            "name": "NIRS",
        },
        {
            "params": list(model.cross_fusion.parameters()) + list(model.cross_hbo.parameters())
            + list(model.cross_hbr.parameters()) + list(model.cross_to_fusion.parameters())
            + list(model.nirs_hb_gate.parameters()) + list(model.cross_hb_gate.parameters())
            + list(model.hbo_feature_adapter.parameters()) + list(model.hbr_feature_adapter.parameters())
            + list(model.cross_logit_head.parameters()) + list(model.hbo_cross_head.parameters())
            + list(model.hbr_cross_head.parameters()) + list(model.logit_gate.parameters())
            + list(model.hb_logit_gate.parameters())
            + [model.aux_logit_scale, model.hb_split_scale, model.hb_logit_scale]
            + list(model.eeg_aux_head.parameters()) + list(model.nirs_aux_head.parameters())
            + list(model.fusion_gate.parameters()) + list(model.classifier.parameters()),
            "lr": args.lr_head,
            "name": "Head",
        },
    ], weight_decay=args.weight_decay)


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = ModerateDataTransform(noise_std=args.noise_std, mask_ratio=args.mask_ratio)
    train_set = ShinMultimodalDataset(
        args.data_path,
        train=True,
        transform=train_transform,
        shuffle_data=args.shuffle_data,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        subject_gap=args.subject_gap,
    )
    test_set = ShinMultimodalDataset(
        args.data_path,
        train=False,
        transform=None,
        shuffle_data=args.shuffle_data,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        subject_gap=args.subject_gap,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    print(f"Dataset Size: Train={len(train_set)}, Test={len(test_set)} | Split:{args.split_mode} Seed:{args.seed}")

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
        nirs_layout=args.nirs_layout,
        nirs_dim_model=args.nirs_dim_model,
        nirs_depth=args.nirs_depth,
        nirs_heads=args.nirs_heads,
        nirs_dropout=args.nirs_dropout,
        classifier_dropout=args.classifier_dropout,
        preblock_residual_scale=args.preblock_residual_scale,
        preblock_mode=args.preblock_mode,
        cross_attn_dropout=args.cross_attn_dropout,
        cross_delay_frames=args.cross_delay_frames,
        cross_window_frames=args.cross_window_frames,
        cross_alignment_mode=args.cross_alignment_mode,
        cross_fusion_mode=args.cross_fusion_mode,
        aux_logit_scale_init=args.aux_logit_scale_init,
        nirs_hb_mode=args.nirs_hb_mode,
        nirs_hb_branch_mode=args.nirs_hb_branch_mode,
        hb_split_scale_init=args.hb_split_scale_init,
        hb_logit_scale_init=args.hb_logit_scale_init,
    ).to(device)

    if args.checkpoint_path.lower() in {"", "none", "null"}:
        print("(!) Training from scratch; no checkpoint loaded.")
    elif os.path.exists(args.checkpoint_path):
        load_partial_checkpoint(model, args.checkpoint_path, device)
    else:
        print(f"Error: Checkpoint {args.checkpoint_path} not found! You must run v13 first.")
        return

    if args.reset_head:
        print("(!) Resetting cross-attention fusion, fusion gate and classifier.")
        reset_module_parameters(model.cross_fusion)
        reset_module_parameters(model.cross_hbo)
        reset_module_parameters(model.cross_hbr)
        reset_module_parameters(model.nirs_hb_encoder)
        reset_module_parameters(model.nirs_hbo_encoder)
        reset_module_parameters(model.nirs_hbr_encoder)
        reset_module_parameters(model.hbo_feature_adapter)
        reset_module_parameters(model.hbr_feature_adapter)
        reset_module_parameters(model.cross_to_fusion)
        reset_module_parameters(model.nirs_hb_gate)
        reset_module_parameters(model.cross_hb_gate)
        model.hb_split_scale.data.fill_(float(args.hb_split_scale_init))
        reset_module_parameters(model.cross_logit_head)
        reset_module_parameters(model.hbo_cross_head)
        reset_module_parameters(model.hbr_cross_head)
        reset_module_parameters(model.logit_gate)
        reset_module_parameters(model.hb_logit_gate)
        model.hb_logit_scale.data.fill_(float(args.hb_logit_scale_init))
        reset_module_parameters(model.eeg_aux_head)
        reset_module_parameters(model.nirs_aux_head)
        reset_module_parameters(model.fusion_gate)
        reset_module_parameters(model.classifier)

    if args.freeze_nirs_encoder:
        print("(!) Freezing original ComplexNIRS_Encoder; training PreBlock + fusion/head only.")
        set_module_trainable(model.nirs_encoder, False)
        set_module_trainable(model.nirs_hb_encoder, False)
        set_module_trainable(model.nirs_hbo_encoder, False)
        set_module_trainable(model.nirs_hbr_encoder, False)

    print("(!) Unfreezing EVERYTHING for Fine-Tuning.")
    for param in model.parameters():
        if param.dtype.is_floating_point:
            param.requires_grad = True
    if args.freeze_nirs_encoder:
        set_module_trainable(model.nirs_encoder, False)
        set_module_trainable(model.nirs_hb_encoder, False)
        set_module_trainable(model.nirs_hbo_encoder, False)
        set_module_trainable(model.nirs_hbr_encoder, False)

    optimizer = build_optimizer(model, args)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    best_acc = 0.0

    for epoch in range(args.epochs):
        in_stage1 = epoch < args.stage1_epochs
        set_module_trainable(model.eeg_encoder, not in_stage1)
        set_module_trainable(model.eeg_adapter, not in_stage1)
        if args.freeze_nirs_encoder:
            set_module_trainable(model.nirs_encoder, False)
            set_module_trainable(model.nirs_hb_encoder, False)
            set_module_trainable(model.nirs_hbo_encoder, False)
            set_module_trainable(model.nirs_hbr_encoder, False)
        if epoch == 0:
            print(f"Stage 1: freeze EEG/adapter for {args.stage1_epochs} epochs; train NIRS + head.")
        if epoch == args.stage1_epochs:
            print("Stage 2: unfreeze EEG/adapter for joint fine-tuning.")

        model.train()
        total_loss, total_main_loss = 0, 0
        total_eeg_aux_loss, total_nirs_aux_loss = 0, 0
        total_hbo_aux_loss, total_hbr_aux_loss = 0, 0
        correct, total = 0, 0
        cur_mixup_alpha = args.stage1_mixup_alpha if in_stage1 else args.mixup_alpha

        for x_eeg, x_nirs, y in train_loader:
            x_eeg, x_nirs, y = x_eeg.to(device), x_nirs.to(device), y.to(device)
            optimizer.zero_grad()
            mixed_x_eeg, mixed_x_nirs, y_a, y_b, lam = mixup_data(
                x_eeg, x_nirs, y, alpha=cur_mixup_alpha, device=device
            )
            use_aux_loss = (
                args.aux_eeg_loss_weight > 0 or args.aux_nirs_loss_weight > 0
                or args.aux_hbo_loss_weight > 0 or args.aux_hbr_loss_weight > 0
            )
            if use_aux_loss:
                logits, logits_eeg, logits_nirs, logits_hbo, logits_hbr = model(
                    mixed_x_eeg, mixed_x_nirs, return_aux=True
                )
                main_loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                eeg_aux_loss = mixup_criterion(criterion, logits_eeg, y_a, y_b, lam)
                nirs_aux_loss = mixup_criterion(criterion, logits_nirs, y_a, y_b, lam)
                hbo_aux_loss = mixup_criterion(criterion, logits_hbo, y_a, y_b, lam)
                hbr_aux_loss = mixup_criterion(criterion, logits_hbr, y_a, y_b, lam)
                loss = (
                    main_loss
                    + args.aux_eeg_loss_weight * eeg_aux_loss
                    + args.aux_nirs_loss_weight * nirs_aux_loss
                    + args.aux_hbo_loss_weight * hbo_aux_loss
                    + args.aux_hbr_loss_weight * hbr_aux_loss
                )
            else:
                logits = model(mixed_x_eeg, mixed_x_nirs)
                main_loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                eeg_aux_loss = logits.new_tensor(0.0)
                nirs_aux_loss = logits.new_tensor(0.0)
                hbo_aux_loss = logits.new_tensor(0.0)
                hbr_aux_loss = logits.new_tensor(0.0)
                loss = main_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_eeg_aux_loss += eeg_aux_loss.item()
            total_nirs_aux_loss += nirs_aux_loss.item()
            total_hbo_aux_loss += hbo_aux_loss.item()
            total_hbr_aux_loss += hbr_aux_loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (
                lam * predicted.eq(y_a.data).cpu().sum().float()
                + (1 - lam) * predicted.eq(y_b.data).cpu().sum().float()
            )

        train_acc = correct / total * 100

        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for x_eeg, x_nirs, y in test_loader:
                x_eeg, x_nirs, y = x_eeg.to(device), x_nirs.to(device), y.to(device)
                t_correct += (model(x_eeg, x_nirs).argmax(1) == y).sum().item()
                t_total += y.size(0)

        test_acc = 100 * t_correct / t_total
        scheduler.step(test_acc)
        lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
        stage_name = "S1" if in_stage1 else "S2"
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] [{stage_name}] "
            f"LR(E):{lrs['EEG']:.1e} LR(A):{lrs['Adapter']:.1e} "
            f"LR(P):{lrs['PreBlock']:.1e} LR(N):{lrs['NIRS']:.1e} "
            f"LR(H):{lrs['Head']:.1e} | Mixup:{cur_mixup_alpha:.2f} "
            f"| Loss:{total_loss / len(train_loader):.4f} "
            f"Main:{total_main_loss / len(train_loader):.4f} "
            f"AuxE:{total_eeg_aux_loss / len(train_loader):.4f} "
            f"AuxN:{total_nirs_aux_loss / len(train_loader):.4f} "
            f"AuxHbO:{total_hbo_aux_loss / len(train_loader):.4f} "
            f"AuxHbR:{total_hbr_aux_loss / len(train_loader):.4f} "
            f"| TrAcc: {train_acc:.2f}% | TeAcc: {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.save_path)
            print("    (Saved Best Finetuned)")
