# NeuroLoad-Fusion

NeuroLoad-Fusion is an EEG-fNIRS multimodal model for three-class cognitive
load recognition. It extends the BIOT EEG encoder with adaptive fNIRS
preprocessing, HbO/HbR-specific adapters, delay-aware token-level
cross-attention, and gated logit-level fusion.

## Model Overview

- **EEG branch:** STFT-based BIOT encoder, channel tokens, positional
  encoding, and a linear-attention Transformer.
- **fNIRS branch:** adaptive multi-scale trend suppression followed by a
  multi-scale convolutional encoder with SE attention.
- **Hb modeling:** a mixed fNIRS stream plus residual HbO/HbR streams with
  shared encoder and modality-specific adapters.
- **Fusion:** fNIRS tokens query temporally delayed EEG tokens; a gated
  auxiliary logit is added to the feature-level classifier output.

The V14 implementation keeps the original model parameter names so compatible
PyTorch `state_dict` checkpoints can be loaded without conversion.

## Repository Layout

```text
model/                 BIOT encoder and fNIRS building blocks
multimodal_v14/        V14 model, data pipeline, training loop, and CLI
run_multimodal_v14.py  Backward-compatible training entry point
run_multimodal_v14_subject_split.py  Subject-level evaluation script
tests/                 Module-split compatibility tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Format

The training entry point expects one `.pkl` file per sample with the following
keys:

```python
{
    "X_eeg": numpy.ndarray,   # (30, 2000)
    "X_nirs": numpy.ndarray,  # (36, 100)
    "y": int,                 # cognitive-load class: 0, 1, or 2
}
```

Offline HRF-prior alignment additionally requires a real EEG history buffer;
it does not synthesize history from the legacy 10-second EEG window:

```python
{
    "X_eeg": numpy.ndarray,          # (30, 2000), current [t, t + 10 s] EEG
    "X_eeg_context": numpy.ndarray,  # (30, 3600), EEG over [t - 8 s, t + 10 s]
    "X_nirs": numpy.ndarray,         # (36, 100), current [t, t + 10 s] fNIRS
    "y": int,
}
```

Use it with `--cross_alignment_mode soft_hrf`. The mode keeps the existing
10-second EEG branch for classification and uses `X_eeg_context` only as
time-safe cross-attention keys and values. `local` remains the legacy
fixed-delay baseline.

For the default `block` fNIRS layout, the first 18 channels are HbO and the
last 18 channels are HbR.

## Training

Train V14 with a subject-independent split:

```bash
python run_multimodal_v14.py \
  --data_path /path/to/processed_samples \
  --checkpoint_path none \
  --split_mode subject \
  --shuffle_data \
  --nirs_hb_mode mixed_split_residual \
  --nirs_hb_branch_mode adapter \
  --cross_alignment_mode local \
  --cross_fusion_mode logit \
  --hb_logit_scale_init 0.02 \
  --save_path checkpoints/neuroload_fusion_v14.pth
```

For offline HRF-prior alignment, add:

```bash
--cross_alignment_mode soft_hrf \
--offline_eeg_history_sec 8 \
--hrf_lag_max_sec 8 \
--hrf_lag_bin_sec 0.5 \
--hrf_prior_mean_sec 5 \
--hrf_prior_std_sec 1.5
```

The script first freezes the EEG encoder for `--stage1_epochs`, then unfreezes
it for joint fine-tuning. Use `--help` to inspect the full configuration.

## Evaluation

Use the predefined train/validation/test subject split:

```bash
python run_multimodal_v14_subject_split.py \
  --data_path /path/to/processed_samples \
  --checkpoint_path /path/to/model.pth \
  --save_dir runs/subject_split
```

Datasets and trained weights are intentionally not distributed in this
repository. Obtain and use them according to their own licenses and consent
requirements.

## Attribution and License

NeuroLoad-Fusion is derived in part from the MIT-licensed
[BIOT](https://github.com/ycq091044/BIOT) implementation. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE).
