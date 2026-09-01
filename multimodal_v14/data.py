import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class ModerateDataTransform:
    def __init__(self, noise_std=0.02, mask_ratio=0.05, shift_max=5):
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.shift_max = shift_max

    def __call__(self, x):
        if self.shift_max > 0:
            shift = np.random.randint(-self.shift_max, self.shift_max)
            x = torch.roll(x, shifts=shift, dims=1)
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        if self.mask_ratio > 0:
            time_steps = x.shape[1]
            mask_len = int(time_steps * self.mask_ratio)
            if mask_len > 0:
                start = np.random.randint(0, time_steps - mask_len)
                x[:, start:start + mask_len] = 0.0
        return x


def mixup_data(x_eeg, x_nirs, y, alpha=0.4, device="cuda"):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x_eeg.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x_eeg = lam * x_eeg + (1 - lam) * x_eeg[index, :]
    mixed_x_nirs = lam * x_nirs + (1 - lam) * x_nirs[index, :]
    y_a, y_b = y, y[index]
    return mixed_x_eeg, mixed_x_nirs, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class ShinMultimodalDataset(Dataset):
    def __init__(self, root_path, train=True, split_ratio=0.8, transform=None,
                 shuffle_data=False, split_mode="file", split_seed=42, subject_gap=0):
        self.files = []
        all_files = [f for f in os.listdir(root_path) if f.endswith(".pkl")]
        if split_mode == "subject":
            subjects = sorted({f.split("_", 1)[0] for f in all_files})
            if shuffle_data:
                random.Random(split_seed).shuffle(subjects)
            split_idx = int(len(subjects) * split_ratio)
            selected_subjects = set(subjects[:split_idx] if train else subjects[split_idx + subject_gap:])
            self.file_list = sorted(f for f in all_files if f.split("_", 1)[0] in selected_subjects)
        else:
            if shuffle_data:
                random.Random(split_seed).shuffle(all_files)
            else:
                all_files.sort()
            split_idx = int(len(all_files) * split_ratio)
            self.file_list = all_files[:split_idx] if train else all_files[split_idx:]
        self.root_path = root_path
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def normalize(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + 1e-5)

    def __getitem__(self, idx):
        path = os.path.join(self.root_path, self.file_list[idx])
        with open(path, "rb") as f:
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
