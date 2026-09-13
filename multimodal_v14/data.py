import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class ModerateDataTransform:
    def __init__(self, noise_std=0.02, mask_ratio=0.05, shift_max=0, time_shift_max_sec=0.0):
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.shift_max = shift_max
        self.time_shift_max_sec = time_shift_max_sec

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

    @staticmethod
    def _shift_without_wrap(x, shift):
        if shift == 0:
            return x
        shifted = torch.zeros_like(x)
        if shift > 0:
            shifted[:, shift:] = x[:, :-shift]
        else:
            shifted[:, :shift] = x[:, -shift:]
        return shifted

    def _apply_noise_and_mask(self, x):
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.mask_ratio > 0:
            time_steps = x.shape[1]
            mask_len = int(time_steps * self.mask_ratio)
            if mask_len > 0:
                start = np.random.randint(0, time_steps - mask_len)
                x[:, start:start + mask_len] = 0.0
        return x

    def apply_pair(self, x_eeg, x_nirs, eeg_sample_rate, nirs_sample_rate, eeg_context=None):
        if self.time_shift_max_sec > 0:
            shift_sec = np.random.uniform(-self.time_shift_max_sec, self.time_shift_max_sec)
            x_eeg = self._shift_without_wrap(x_eeg, int(round(shift_sec * eeg_sample_rate)))
            x_nirs = self._shift_without_wrap(x_nirs, int(round(shift_sec * nirs_sample_rate)))
            if eeg_context is not None:
                eeg_context = self._shift_without_wrap(eeg_context, int(round(shift_sec * eeg_sample_rate)))

        x_eeg = self._apply_noise_and_mask(x_eeg)
        x_nirs = self._apply_noise_and_mask(x_nirs)
        if eeg_context is None:
            return x_eeg, x_nirs
        return x_eeg, x_nirs, self._apply_noise_and_mask(eeg_context)


def mixup_data(x_eeg, x_nirs, y, alpha=0.4, device="cuda", return_index=False):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x_eeg.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x_eeg = lam * x_eeg + (1 - lam) * x_eeg[index, :]
    mixed_x_nirs = lam * x_nirs + (1 - lam) * x_nirs[index, :]
    y_a, y_b = y, y[index]
    if return_index:
        return mixed_x_eeg, mixed_x_nirs, y_a, y_b, lam, index
    return mixed_x_eeg, mixed_x_nirs, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class ShinMultimodalDataset(Dataset):
    def __init__(self, root_path, train=True, split_ratio=0.8, transform=None,
                 shuffle_data=False, split_mode="file", split_seed=42, subject_gap=0,
                 eeg_context_key=None, expected_eeg_context_shape=None,
                 eeg_sample_rate=200.0, nirs_sample_rate=10.0):
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
        self.eeg_context_key = eeg_context_key
        self.expected_eeg_context_shape = expected_eeg_context_shape
        self.eeg_sample_rate = eeg_sample_rate
        self.nirs_sample_rate = nirs_sample_rate

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
        eeg_context = None
        if self.eeg_context_key is not None:
            if self.eeg_context_key not in data:
                raise KeyError(f"Missing required EEG context key '{self.eeg_context_key}' in {path}.")
            eeg_context = torch.from_numpy(data[self.eeg_context_key]).float()
            if self.expected_eeg_context_shape is not None and tuple(eeg_context.shape) != tuple(self.expected_eeg_context_shape):
                raise ValueError(
                    f"Expected {self.eeg_context_key} shape {self.expected_eeg_context_shape}, "
                    f"got {tuple(eeg_context.shape)} in {path}."
                )
            eeg_context = self.normalize(eeg_context)
        if self.transform:
            transformed = self.transform.apply_pair(
                x_eeg,
                x_nirs,
                eeg_sample_rate=self.eeg_sample_rate,
                nirs_sample_rate=self.nirs_sample_rate,
                eeg_context=eeg_context,
            )
            if eeg_context is None:
                x_eeg, x_nirs = transformed
            else:
                x_eeg, x_nirs, eeg_context = transformed
        if eeg_context is not None:
            return x_eeg, x_nirs, eeg_context, y
        return x_eeg, x_nirs, y


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
