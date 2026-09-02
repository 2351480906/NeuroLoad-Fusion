# NeuroLoad-Fusion EEG Pretraining Design

Date: 2026-09-02

## 1. Objective

Add an EEG-only supervised pretraining stage before multimodal V14 fusion
fine-tuning. The stage uses the Cognitive Workload and Executive Functioning
datasets, preserves the existing V14 `state_dict` contract, and does not depend
on the future fNIRS pretraining pipeline.

The EEG stage will:

1. Convert both source datasets into a common 28-channel, 200 Hz, 10-second
   window format.
2. Train the existing V14 EEG path with a three-class cognitive-load target.
3. Produce a checkpoint that can strictly replace only the EEG-related V14
   parameters.
4. Validate on held-out subjects from both source datasets before any fusion
   work begins.

This design does not implement fNIRS pretraining or run multimodal fusion
training.

## 2. Compatibility Constraints

The following V14 module names and parameter keys must remain unchanged:

- `eeg_encoder.*`
- `eeg_adapter.*`
- `eeg_aux_head.*`

`BIOTEncoder` remains configured with `n_channels=30`, so
`eeg_encoder.channel_tokens.weight` keeps its existing shape. All new EEG
training and future fusion fine-tuning will pass only the 28 scalp EEG channels.
The two rows formerly associated with `HEOG` and `VEOG` remain present for
checkpoint compatibility but are not used.

No existing V14 module is moved or renamed. The EEG pretraining checkpoint is
loaded by an explicit prefix whitelist; non-EEG V14 parameters must remain
byte-for-byte unchanged during the overlay operation.

## 3. Source Datasets and Labels

### 3.1 Cognitive Workload

Source directory:

`/mnt/data/wrf/EEGfNIRS/datasets/Cognitive Workload`

Use the 18 released subjects. Subjects `sub-013` and `sub-017` are excluded by
the source dataset and have no released EEG recordings.

Only formal task events with `istutorial=false` are eligible. Keep levels 1,
2, and 3; exclude tutorial events, 4-back, marker events, and dropout events as
class targets.

### 3.2 Executive Functioning

Source directory:

`/mnt/data/wrf/EEGfNIRS/datasets/Executive Functioning`

Use all 24 subjects, both `ses-pre` and `ses-post`, and only task files `NB1`,
`NB2`, and `NB3`. Exclude SART and Local Global recordings. A recording remains
eligible when it contains fewer than the intended 280 stimuli, provided it has
at least one complete valid 10-second window after cropping and quality checks.

### 3.3 Common Label Mapping

| Source condition | Training label | Meaning | Proxy |
| --- | ---: | --- | --- |
| 1-back / NB1 | 0 | low load | yes |
| 2-back / NB2 | 1 | medium load | no |
| 3-back / NB3 | 2 | high load | no |

Every sample records both `mapped_label` and `source_nback`. A 1-back sample is
never described as a literal 0-back sample in metadata or reports.

## 4. Canonical EEG Representation

The canonical channel order is the first 28 channels of the current
multimodal EEG representation:

```text
Fp1, AFF5h, AFz, F1, FC5, FC1, T7, C3, Cz, CP5, CP1, P7, P3, Pz,
POz, O1, Fp2, AFF6h, F2, FC2, FC6, C4, T8, CP2, CP6, P4, P8, O2
```

Legacy aliases are normalized before montage assignment:

```text
T3 -> T7
T4 -> T8
T5 -> P7
T6 -> P8
FP1 -> Fp1
FP2 -> Fp2
```

The source recording keeps all usable scalp EEG channels during interpolation,
including channels that are not in the final 28-channel list. Missing target
channels are added at standard 10-20/10-05 coordinates, marked bad, and
interpolated from the usable source channels. After interpolation, only the 28
canonical channels are selected in the order above.

No ECG, MISC, HEOG, VEOG, or synthetic EOG channels are included.

## 5. Signal Preprocessing

Preprocessing is deterministic and does not use ICA.

For each recording:

1. Read the continuous signal and BIDS sidecars with MNE.
2. Normalize channel names and retain scalp EEG channels only.
3. Attach the standard montage and reject duplicate or unresolved required
   channel names.
4. Apply a zero-phase 1-45 Hz band-pass filter.
5. Detect unusable source channels using non-finite data, a continuous
   five-second flatline (consecutive absolute sample differences no greater
   than `1e-12` volts), or an absolute robust log-variance z-score greater
   than 5 across source channels. The robust z-score uses the channel median
   and median absolute deviation.
6. Abort the recording if more than 25% of its source EEG channels are unusable
   or fewer than 12 usable spatial channels remain.
7. Add and interpolate missing canonical channels using spherical spline
   interpolation.
8. Select the canonical 28 channels and apply a 28-channel average reference.
9. Resample to 200 Hz using anti-alias filtering.
10. Generate candidate windows using the dataset-specific rules below.
11. Store accepted windows as `float32`; defer per-window, per-channel
    standardization to the training dataset loader.

The Executive Functioning README and conversion metadata disagree about prior
filtering and referencing. The common pipeline therefore treats its BIDS files
as not yet harmonized and applies the same deterministic processing used for
Cognitive Workload.

## 6. Window Generation and Quality Control

All accepted model inputs have shape `(28, 2000)`. Window length and stride are
both 10 seconds; overlapping windows are not generated.

### 6.1 Cognitive Workload windows

For each formal 1-, 2-, or 3-back block:

- Block start is the first formal trial onset plus a five-second guard.
- Block end is the last formal trial onset plus 1.7 seconds, minus a
  five-second guard.
- Generate complete 10-second windows wholly contained in this interval.
- Label every window by the enclosing n-back block, regardless of response
  correctness.
- Sum dropout samples whose annotated intervals overlap the window. Reject the
  window when the total exceeds 2% of the original 250 Hz samples, equivalent
  to more than 50 missing samples in 10 seconds.

### 6.2 Executive Functioning windows

For each NB recording:

- Eligible stimulus events have values A, B, C, or D.
- Task start is the first eligible stimulus onset plus a five-second guard.
- Task end is the last eligible stimulus onset plus 1.5 seconds, minus a
  five-second guard.
- Generate complete non-overlapping 10-second windows inside that interval.
- Label every window from the file task (`NB1`, `NB2`, or `NB3`), regardless of
  target status or response correctness.

### 6.3 Common window rejection

Reject a candidate window when any of the following holds:

- It contains NaN or infinity.
- More than 20% of canonical channels have peak-to-peak amplitude below
  0.5 microvolts or above 500 microvolts.
- Any channel contains a continuous two-second digital flatline, defined as
  consecutive absolute sample differences no greater than `1e-12` volts.
- Its final shape differs from `(28, 2000)`.

Every rejection records a machine-readable reason. No sample is silently
dropped.

## 7. Processed Data Contract

The output directory is supplied explicitly by CLI. Data is stored per subject
as memory-mappable NumPy arrays:

```text
processed_root/
|-- manifest.jsonl
|-- splits.json
|-- qc_summary.json
`-- subjects/
    |-- cognitive_workload/<subject_id>/
    |   |-- X.npy
    |   |-- y.npy
    |   `-- metadata.jsonl
    `-- executive_functioning/<subject_id>/
        |-- X.npy
        |-- y.npy
        `-- metadata.jsonl
```

Each metadata row includes:

- dataset and source file
- subject and session
- source n-back level and mapped label
- `is_proxy_label`
- source and resampled start/end times
- channel order and preprocessing version
- quality-control values

`manifest.jsonl` indexes each subject array and row without duplicating signal
data. Subject outputs are written to a temporary sibling path and atomically
renamed only after their arrays and metadata validate successfully.

## 8. Subject Splits and Sampling

Splits are generated before training and saved in `splits.json`.

### 8.1 Cognitive Workload

- 12 train subjects
- 3 validation subjects
- 3 test subjects

Sort subject IDs, shuffle with seed 42, then take train, validation, and test in
that order.

### 8.2 Executive Functioning

- 16 train subjects
- 4 validation subjects
- 4 test subjects

Split independently within the `FB` and `S` groups so each group contributes
8 train, 2 validation, and 2 test subjects. A subject's pre/post sessions always
remain in the same split. The group lists are sorted and shuffled
deterministically from seed 42.

Training uses a weighted sampler with per-sample weight:

```text
weight(sample) = 1 / sqrt(N[dataset, mapped_label])
```

Validation and test loaders do not resample or weight examples.

## 9. EEG Pretraining Model

The EEG-only model contains the existing V14 modules under their original
attribute names:

```text
x: (B, 28, 2000)
  -> eeg_encoder.forward_tokens(x)
  -> mean over tokens
  -> eeg_adapter
  -> eeg_aux_head
  -> logits: (B, 3)
```

The model is initialized by strictly extracting `eeg_encoder.*`,
`eeg_adapter.*`, and `eeg_aux_head.*` from:

`/mnt/data/wrf/BIOT/reliable_subjectgap_seed42_v14_hbadapter_hblogit002.pth`

The known V14 architecture configuration remains unchanged. This experiment is
therefore external EEG continued pretraining, not training from random weights.

## 10. Training Procedure

Default training configuration:

| Setting | Value |
| --- | ---: |
| seed | 42 |
| batch size | 32 |
| maximum epochs | 50 |
| frozen encoder epochs | 5 |
| encoder learning rate | 1e-5 |
| adapter/head learning rate | 1e-4 |
| optimizer | AdamW |
| weight decay | 1e-4 |
| label smoothing | 0.05 |
| mixup alpha | 0.2 |
| early stopping patience | 10 |

During the first five epochs, only `eeg_adapter` and `eeg_aux_head` are
trainable. The complete EEG path is then unfrozen, retaining differential
learning rates. `ReduceLROnPlateau` monitors the checkpoint-selection metric.

Training inputs are normalized independently per channel and per window using
the same epsilon-stabilized standardization as the current V14 dataset. Mixup
is applied only to training batches.

The checkpoint-selection metric is:

```text
(Cognitive Workload validation macro-F1
 + Executive Functioning validation macro-F1) / 2
```

Training also reports accuracy, macro precision, macro recall, macro-F1, and a
confusion matrix separately for each dataset and for the pooled examples.

## 11. Checkpoint Contract and V14 Overlay

The best and final checkpoints contain:

- format version
- model state dictionary
- source V14 checkpoint path and configuration
- canonical channel order
- preprocessing version and label mapping
- subject split lists
- optimizer-independent training configuration
- best epoch and validation/test metrics

The V14 overlay loader accepts only these prefixes:

```text
eeg_encoder.
eeg_adapter.
eeg_aux_head.
```

It fails when a whitelisted key is missing, unexpected, duplicated, or has a
different shape. A compatibility check snapshots every non-EEG V14 tensor,
applies the overlay, and verifies that all snapshots are unchanged.

Future fusion fine-tuning will instantiate V14 with the existing 30-channel
parameter capacity but pass only the canonical 28 EEG channels. Fusion
fine-tuning itself is outside this EEG-only implementation.

## 12. Code Organization

Add:

```text
eeg_pretraining/
|-- __init__.py
|-- data.py
|-- preprocessing.py
|-- modeling.py
|-- training.py
`-- cli.py

prepare_eeg_pretraining.py
run_eeg_pretraining.py
tests/test_eeg_pretraining.py
```

Update:

- `requirements.txt` with `mne` and `scipy`, which are required to read and
  process the BIDS BrainVision and EEGLAB files.
- `README.md` with preprocessing, audit, and training commands.

The existing `multimodal_v14` and `model` module ownership boundaries remain
unchanged.

## 13. Error Handling

The preprocessing command operates in strict mode:

- Missing mandatory files or sidecars fail with dataset, subject, session, and
  path context.
- Unknown task labels or channel aliases fail instead of being guessed.
- A recording with too few usable channels or no valid windows fails.
- Incomplete Executive Functioning recordings are accepted only when all
  remaining windows satisfy the normal rules.
- Partial per-subject outputs are never promoted to final output paths.
- The command exits nonzero when any required subject fails and retains a QC
  report describing the failure.

Training validates the manifest, split disjointness, class coverage, array
shape, finite values, and checkpoint compatibility before creating an
optimizer.

## 14. Verification and Acceptance Criteria

Unit tests use synthetic temporary data and cover:

1. Label mapping and exclusion of tutorial, 4-back, SART, and LG events.
2. Legacy channel aliases and exact canonical channel order.
3. Dataset-specific task boundaries, five-second guards, 10-second windows,
   and dropout threshold behavior.
4. Window QC rejection reasons.
5. Exact 12/3/3 and 16/4/4 subject splits with no overlap and grouped pre/post
   sessions.
6. Square-root sample weights.
7. EEG forward output shape for `(B, 28, 2000)` while the encoder retains a
   30-row channel embedding table.
8. Strict extraction from a synthetic V14 checkpoint with the production key
   layout.
9. Strict EEG overlay into V14 and equality of every non-EEG tensor before and
   after overlay.
10. A one-batch CPU training smoke test.

A separate local integration check uses the known V14 checkpoint path to
verify real checkpoint extraction and overlay. The ordinary unit test suite
does not require that external 23 MB file.

Before full preprocessing, run a one-subject dry run from each dataset and
inspect the channel mapping, event intervals, sample shapes, and QC summary.
Full preprocessing may proceed only after both dry runs pass.

Before full training, inspect `qc_summary.json` for per-dataset, per-subject,
and per-class accepted/rejected counts. Long-running training starts only after
that audit is accepted.

## 15. Known Limitation

The existing V14 checkpoint was selected using test accuracy in the original
training flow. Reusing it as the EEG initialization preserves the requested
continuity but is not equivalent to a target-independent pretrained model.
Results must be described as continued pretraining, and unbiased final claims
require target subjects that were not used to select the original checkpoint.
