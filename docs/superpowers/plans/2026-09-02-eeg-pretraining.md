# EEG Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, and run an EEG-only continued-pretraining pipeline for Cognitive Workload and Executive Functioning data while preserving the V14 EEG `state_dict` contract.

**Architecture:** A new `eeg_pretraining` package owns deterministic BIDS preprocessing, memory-mapped subject datasets, the EEG-only V14 path, strict checkpoint transfer, and staged training. Existing `model` and `multimodal_v14` modules remain structurally unchanged; the pretraining model reuses their modules under the exact keys `eeg_encoder`, `eeg_adapter`, and `eeg_aux_head`.

**Tech Stack:** Python 3, PyTorch, NumPy, MNE, SciPy, scikit-learn, standard-library `unittest`

**Spec:** `docs/superpowers/specs/2026-09-02-eeg-pretraining-design.md`

## Global Constraints

- Keep `BIOTEncoder(n_channels=30)` and pass only the canonical 28 scalp EEG channels.
- Do not rename, move, or reshape any existing V14 parameter-bearing module.
- Transfer only `eeg_encoder.*`, `eeg_adapter.*`, and `eeg_aux_head.*`.
- Instantiate the source V14 checkpoint with EEG `(emb_size=256, heads=8, depth=4, n_channels=30, n_fft=200, hop_length=100)` and the approved `mixed_split_residual` / `adapter` / `local` / `logit` / `hb_logit_scale_init=0.02` configuration.
- Map 1-back to label 0 as a documented proxy, 2-back to 1, and 3-back to 2; exclude 4-back and tutorial data.
- Use the exact canonical channel order from the design spec.
- Use 1-45 Hz filtering, average reference, 200 Hz output, 10-second non-overlapping windows, and no ICA.
- Split by subject before training; keep Executive Functioning pre/post sessions together.
- Do not start full preprocessing until both one-subject dry runs pass.
- Do not start long-running training until the full preprocessing QC report is reviewed.
- Preserve unrelated user changes and keep each implementation commit scoped to one task.

## File Map

| File | Responsibility |
| --- | --- |
| `eeg_pretraining/__init__.py` | Stable public exports for preprocessing, data, model, and checkpoint APIs |
| `eeg_pretraining/preprocessing.py` | Channel rules, source discovery, event intervals, signal harmonization, window QC, atomic subject output |
| `eeg_pretraining/data.py` | Deterministic subject splits, manifest validation, mmap dataset, normalization, weighted sampler |
| `eeg_pretraining/modeling.py` | EEG-only model and strict extraction/overlay of V14 EEG parameters |
| `eeg_pretraining/training.py` | Mixup, staged freezing, optimizer groups, metrics, early stopping, checkpoint serialization |
| `eeg_pretraining/cli.py` | Parsers and argument-to-config conversion for preprocessing and training |
| `prepare_eeg_pretraining.py` | Thin preprocessing entry point |
| `run_eeg_pretraining.py` | Thin training and checkpoint-validation entry point |
| `tests/test_eeg_pretraining.py` | Synthetic unit and smoke tests for every public contract |
| `requirements.txt` | Add MNE and SciPy runtime dependencies |
| `README.md` | Document dry run, full preprocessing, validation-only, and training commands |

---

### Task 1: Core EEG Metadata and Channel Contracts

**Files:**
- Create: `eeg_pretraining/__init__.py`
- Create: `eeg_pretraining/preprocessing.py`
- Modify: `requirements.txt`
- Create: `tests/test_eeg_pretraining.py`

**Interfaces:**
- Produces: `CANONICAL_EEG_CHANNELS: tuple[str, ...]`
- Produces: `CHANNEL_ALIASES: dict[str, str]`
- Produces: `PreprocessingError(RuntimeError)`
- Produces: `map_source_nback(source_nback: int) -> tuple[int, bool]`
- Produces: `canonicalize_channel_name(name: str) -> str`
- Produces: `TaskInterval(dataset: str, subject_id: str, session: str | None, source_nback: int, start_seconds: float, end_seconds: float, source_file: str)`

- [ ] **Step 1: Add failing tests for labels and channel names**

```python
class CoreContractTests(unittest.TestCase):
    def test_label_mapping_marks_only_one_back_as_proxy(self):
        self.assertEqual(map_source_nback(1), (0, True))
        self.assertEqual(map_source_nback(2), (1, False))
        self.assertEqual(map_source_nback(3), (2, False))
        with self.assertRaisesRegex(ValueError, "Unsupported n-back level: 4"):
            map_source_nback(4)

    def test_channel_contract_has_exact_v14_scalp_order(self):
        self.assertEqual(len(CANONICAL_EEG_CHANNELS), 28)
        self.assertEqual(CANONICAL_EEG_CHANNELS[:4], ("Fp1", "AFF5h", "AFz", "F1"))
        self.assertEqual(CANONICAL_EEG_CHANNELS[-4:], ("CP6", "P4", "P8", "O2"))
        self.assertEqual(canonicalize_channel_name("T3"), "T7")
        self.assertEqual(canonicalize_channel_name("FP2"), "Fp2")
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run: `python -m unittest tests.test_eeg_pretraining.CoreContractTests -v`

Expected: FAIL because `eeg_pretraining.preprocessing` does not exist.

- [ ] **Step 3: Add dependencies and the minimal public contracts**

Append `mne` and `scipy` to `requirements.txt`. In `preprocessing.py`, define the exact tuple, aliases, frozen dataclass, and strict mapping:

```python
CANONICAL_EEG_CHANNELS = (
    "Fp1", "AFF5h", "AFz", "F1", "FC5", "FC1", "T7", "C3", "Cz",
    "CP5", "CP1", "P7", "P3", "Pz", "POz", "O1", "Fp2", "AFF6h",
    "F2", "FC2", "FC6", "C4", "T8", "CP2", "CP6", "P4", "P8", "O2",
)

CHANNEL_ALIASES = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "FP1": "Fp1", "FP2": "Fp2",
}

def map_source_nback(source_nback: int) -> tuple[int, bool]:
    mapping = {1: (0, True), 2: (1, False), 3: (2, False)}
    try:
        return mapping[source_nback]
    except KeyError as exc:
        raise ValueError(f"Unsupported n-back level: {source_nback}") from exc
```

Export these symbols from `eeg_pretraining/__init__.py`.

- [ ] **Step 4: Run the focused tests**

Run: `python -m unittest tests.test_eeg_pretraining.CoreContractTests -v`

Expected: PASS.

- [ ] **Step 5: Commit the core contracts**

```bash
git add requirements.txt eeg_pretraining/__init__.py eeg_pretraining/preprocessing.py tests/test_eeg_pretraining.py
git commit -m "feat: add EEG pretraining channel contracts"
```

---

### Task 2: Event Intervals, Windows, and Quality Control

**Files:**
- Modify: `eeg_pretraining/preprocessing.py`
- Modify: `tests/test_eeg_pretraining.py`

**Interfaces:**
- Consumes: `TaskInterval`, `map_source_nback()` from Task 1
- Produces: `build_cognitive_workload_intervals(rows: Sequence[Mapping[str, str]], subject_id: str, source_file: str) -> list[TaskInterval]`
- Produces: `build_executive_functioning_interval(rows: Sequence[Mapping[str, str]], subject_id: str, session: str, source_nback: int, source_file: str) -> TaskInterval`
- Produces: `generate_window_bounds(interval: TaskInterval, sfreq: float = 200.0, window_seconds: float = 10.0) -> list[tuple[int, int]]`
- Produces: `WindowQC(accepted: bool, reasons: tuple[str, ...], metrics: dict[str, float])`
- Produces: `evaluate_window_qc(window: np.ndarray, dropout_samples: int = 0) -> WindowQC`

- [ ] **Step 1: Add failing interval and QC tests**

```python
def test_cognitive_intervals_exclude_tutorial_and_four_back(self):
    rows = [
        {"onset": "10", "trial_type": "1-back", "nback_level": "1", "istutorial": "true"},
        {"onset": "100", "trial_type": "1-back", "nback_level": "1", "istutorial": "false"},
        {"onset": "269", "trial_type": "1-back", "nback_level": "1", "istutorial": "false"},
        {"onset": "300", "trial_type": "4-back", "nback_level": "4", "istutorial": "false"},
    ]
    intervals = build_cognitive_workload_intervals(rows, "sub-001", "events.tsv")
    self.assertEqual([(x.source_nback, x.start_seconds, x.end_seconds) for x in intervals], [(1, 105.0, 265.7)])

def test_window_qc_rejects_more_than_two_percent_dropout(self):
    window = np.linspace(-10e-6, 10e-6, 2000, dtype=np.float64)[None, :]
    window = np.repeat(window, 28, axis=0)
    self.assertTrue(evaluate_window_qc(window, dropout_samples=50).accepted)
    self.assertIn("dropout_ratio", evaluate_window_qc(window, dropout_samples=51).reasons)
```

Also test the Executive Functioning A/B/C/D event filter, exact five-second guards, 10-second non-overlap, NaN rejection, flatline rejection, amplitude rejection, and shape rejection.

- [ ] **Step 2: Run the interval/QC tests and verify failure**

Run: `python -m unittest tests.test_eeg_pretraining.IntervalAndQCTests -v`

Expected: FAIL because interval and QC functions are missing.

- [ ] **Step 3: Implement deterministic interval and QC functions**

Use a five-second guard on both boundaries. For Cognitive Workload, group only formal rows by levels 1-3 and calculate `first_onset + 5.0` and `last_onset + 1.7 - 5.0`. For Executive Functioning, filter values A-D and calculate `first_onset + 5.0` and `last_onset + 1.5 - 5.0`.

Implement QC with explicit reasons:

```python
def evaluate_window_qc(window: np.ndarray, dropout_samples: int = 0) -> WindowQC:
    reasons: list[str] = []
    if window.shape != (28, 2000):
        reasons.append("shape")
    if not np.isfinite(window).all():
        reasons.append("non_finite")
    if dropout_samples > 50:
        reasons.append("dropout_ratio")
    if window.shape == (28, 2000) and np.isfinite(window).all():
        peak_to_peak_uv = np.ptp(window, axis=1) * 1e6
        abnormal = (peak_to_peak_uv < 0.5) | (peak_to_peak_uv > 500.0)
        if float(abnormal.mean()) > 0.20:
            reasons.append("amplitude")
        if _has_flatline(window, samples=400, atol_volts=1e-12):
            reasons.append("flatline")
    return WindowQC(not reasons, tuple(reasons), {"dropout_samples": float(dropout_samples)})
```

- [ ] **Step 4: Run interval/QC tests**

Run: `python -m unittest tests.test_eeg_pretraining.IntervalAndQCTests -v`

Expected: PASS.

- [ ] **Step 5: Commit interval and QC behavior**

```bash
git add eeg_pretraining/preprocessing.py tests/test_eeg_pretraining.py
git commit -m "feat: add EEG window and quality rules"
```

---

### Task 3: Deterministic Subject Splits and Sampling Weights

**Files:**
- Create: `eeg_pretraining/data.py`
- Modify: `eeg_pretraining/__init__.py`
- Modify: `tests/test_eeg_pretraining.py`

**Interfaces:**
- Produces: `build_subject_splits(cognitive_subjects: Sequence[str], executive_groups: Mapping[str, Sequence[str]], seed: int = 42) -> dict[str, dict[str, list[str]]]`
- Produces: `validate_subject_splits(splits: Mapping[str, Mapping[str, Sequence[str]]]) -> None`
- Produces: `compute_sample_weights(rows: Sequence[Mapping[str, object]]) -> np.ndarray`

- [ ] **Step 1: Add failing split and weight tests**

```python
def test_subject_splits_have_exact_sizes_and_no_overlap(self):
    cw = [f"sub-{i:03d}" for i in range(1, 19)]
    groups = {
        "FB": [f"sub-fb-{i:02d}" for i in range(12)],
        "S": [f"sub-s-{i:02d}" for i in range(12)],
    }
    splits = build_subject_splits(cw, groups, seed=42)
    self.assertEqual([len(splits["cognitive_workload"][x]) for x in ("train", "val", "test")], [12, 3, 3])
    self.assertEqual([len(splits["executive_functioning"][x]) for x in ("train", "val", "test")], [16, 4, 4])
    validate_subject_splits(splits)

def test_sample_weights_use_dataset_label_square_root_counts(self):
    rows = [
        {"dataset": "cw", "mapped_label": 0},
        {"dataset": "cw", "mapped_label": 0},
        {"dataset": "ef", "mapped_label": 0},
    ]
    weights = compute_sample_weights(rows)
    np.testing.assert_allclose(weights, [1 / np.sqrt(2), 1 / np.sqrt(2), 1.0])
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m unittest tests.test_eeg_pretraining.SplitAndSamplingTests -v`

Expected: FAIL because `eeg_pretraining.data` does not exist.

- [ ] **Step 3: Implement exact split and sampling contracts**

Sort IDs before shuffling with `np.random.default_rng(seed)`. For Executive Functioning, shuffle each sorted group independently from child RNGs derived from `np.random.SeedSequence(seed)`, then allocate 8/2/2 per group. Validate exact disjointness within and across splits.

Calculate weights with a `Counter((dataset, mapped_label))` and return `float64` weights in input-row order.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_eeg_pretraining.SplitAndSamplingTests -v`

Expected: PASS.

- [ ] **Step 5: Commit subject split logic**

```bash
git add eeg_pretraining/data.py eeg_pretraining/__init__.py tests/test_eeg_pretraining.py
git commit -m "feat: add subject-safe EEG data splits"
```

---

### Task 4: MNE Signal Harmonization

**Files:**
- Modify: `eeg_pretraining/preprocessing.py`
- Modify: `tests/test_eeg_pretraining.py`

**Interfaces:**
- Consumes: canonical channels and aliases from Task 1
- Produces: `detect_bad_source_channels(raw: mne.io.BaseRaw) -> list[str]`
- Produces: `harmonize_raw(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, dict[str, object]]`

- [ ] **Step 1: Add a failing synthetic harmonization test**

Build a 19-channel `mne.io.RawArray` at 250 Hz with standard positions, one legacy channel name, and ten seconds of deterministic 10 Hz signal plus small noise. Assert that harmonization returns exactly the canonical 28 channels at 200 Hz, with finite data and no EOG channels.

```python
harmonized, report = harmonize_raw(raw)
self.assertEqual(harmonized.ch_names, list(CANONICAL_EEG_CHANNELS))
self.assertEqual(harmonized.info["sfreq"], 200.0)
self.assertTrue(np.isfinite(harmonized.get_data()).all())
self.assertEqual(report["output_channels"], list(CANONICAL_EEG_CHANNELS))
```

Add separate tests that a five-second flat source channel is marked bad, duplicate aliases such as simultaneous `T3` and `T7` are rejected, unresolved required montage coordinates are rejected, and more than 25% bad channels raises `PreprocessingError` with the recording identifier.

- [ ] **Step 2: Run harmonization tests and verify failure**

Run: `python -m unittest tests.test_eeg_pretraining.HarmonizationTests -v`

Expected: FAIL because harmonization functions are missing.

- [ ] **Step 3: Implement filtering, bad-channel detection, interpolation, reference, and resampling**

Use `mne.channels.make_standard_montage("standard_1005")`. Compute robust log-variance z-scores with median and MAD. Apply the montage to renamed source channels, add absent canonical channels as zero-valued MNE channels at the correct sampling rate, reapply the montage so the added channels receive coordinates, mark them bad, and call `interpolate_bads(reset_bads=True)`. Select canonical channels, apply average reference, then resample to 200 Hz.

Return a report containing source channels, renamed channels, detected bad channels, interpolated target channels, input/output sampling rates, and output order.

- [ ] **Step 4: Run harmonization tests**

Run: `python -m unittest tests.test_eeg_pretraining.HarmonizationTests -v`

Expected: PASS with no interactive plotting or downloads.

- [ ] **Step 5: Commit harmonization**

```bash
git add eeg_pretraining/preprocessing.py tests/test_eeg_pretraining.py
git commit -m "feat: harmonize EEG recordings to V14 scalp channels"
```

---

### Task 5: Source Discovery and Atomic Preprocessing Outputs

**Files:**
- Modify: `eeg_pretraining/preprocessing.py`
- Modify: `tests/test_eeg_pretraining.py`

**Interfaces:**
- Produces: `SourceRecording(dataset: str, subject_id: str, session: str | None, source_nback: int | None, signal_path: Path, events_path: Path, sidecar_paths: tuple[Path, ...], executive_group: str | None)`
- Produces: `PreprocessingConfig(cognitive_root: Path, executive_root: Path, output_dir: Path, seed: int = 42, limit_subjects_per_dataset: int | None = None)`
- Produces: `discover_cognitive_workload(config: PreprocessingConfig) -> list[SourceRecording]`
- Produces: `discover_executive_functioning(config: PreprocessingConfig) -> list[SourceRecording]`
- Produces: `prepare_eeg_data(config: PreprocessingConfig) -> dict[str, object]`

- [ ] **Step 1: Add failing discovery and atomic-output tests**

Create temporary BIDS-like metadata trees for all 18 Cognitive Workload and 24 Executive Functioning subjects so the production split contract can be built. Set `limit_subjects_per_dataset=1`, mock the binary readers to return synthetic MNE raw objects only for the two selected subjects, then assert:

```python
summary = prepare_eeg_data(config)
self.assertTrue((output / "manifest.jsonl").is_file())
self.assertTrue((output / "splits.json").is_file())
self.assertTrue((output / "qc_summary.json").is_file())
summary_json = json.loads((output / "qc_summary.json").read_text())
self.assertTrue(summary_json["partial"])
X = np.load(output / "subjects" / "cognitive_workload" / "sub-001" / "X.npy", mmap_mode="r")
self.assertEqual(X.shape[1:], (28, 2000))
```

Assert every manifest row contains `dataset`, `subject_id`, `session`, `source_nback`, `mapped_label`, `is_proxy_label`, `x_path`, `y_path`, and `row_index`. Add failure tests that an existing output directory is refused before reading source signals and that an invalid subject leaves no promoted subject directory; the raised `PreprocessingError` must name dataset, subject, session, and source path.

- [ ] **Step 2: Run preprocessing-output tests and verify failure**

Run: `python -m unittest tests.test_eeg_pretraining.PreprocessingOutputTests -v`

Expected: FAIL because discovery and output orchestration are missing.

- [ ] **Step 3: Implement strict discovery and per-subject atomic writes**

Read TSV rows with `csv.DictReader`. Use `mne.io.read_raw_brainvision(..., preload=True)` for Cognitive Workload and `mne.io.read_raw_eeglab(..., preload=True)` for Executive Functioning. Require all documented sidecars and reject unknown tasks. Refuse to run when `output_dir` already exists so a retry cannot mix old and new shards.

Always discover and validate the complete 18/24-subject source inventory and build the production split lists before applying `limit_subjects_per_dataset`. The limit controls only which sorted subjects are materialized. Mark `qc_summary.json` with `partial=true` when a limit is used and list both discovered and materialized subject IDs; partial output is diagnostic data and cannot be used for training.

Build the entire requested output under a unique temporary sibling of `output_dir`. For each materialized subject, write `X.npy`, `y.npy`, and `metadata.jsonl` into a temporary subject directory inside that staging root. Re-open arrays with `mmap_mode="r"`, validate row counts and shapes, then call `os.replace(temp_subject_dir, final_subject_dir)`. Write one manifest row per sample using relative shard paths plus `row_index`. Aggregate accepted and rejected counts by dataset, subject, source level, mapped label, and rejection reason.

After all requested subjects and global files validate, atomically rename the staging root to `output_dir`. On any failure, write an adjacent `<output_dir.name>.failed_qc.json` report, remove only the staging tree created by the current invocation, and leave `output_dir` absent. Tests must cover failure after one valid subject to prove no partially complete dataset is promoted.

- [ ] **Step 4: Run preprocessing-output tests**

Run: `python -m unittest tests.test_eeg_pretraining.PreprocessingOutputTests -v`

Expected: PASS.

- [ ] **Step 5: Commit preprocessing orchestration**

```bash
git add eeg_pretraining/preprocessing.py tests/test_eeg_pretraining.py
git commit -m "feat: prepare EEG subject shards atomically"
```

---

### Task 6: Memory-Mapped Training Dataset

**Files:**
- Modify: `eeg_pretraining/data.py`
- Modify: `eeg_pretraining/__init__.py`
- Modify: `tests/test_eeg_pretraining.py`

**Interfaces:**
- Consumes: `manifest.jsonl` and `splits.json` from Task 5
- Produces: `EEGWindowDataset(processed_root: str | Path, split: str | None, dataset_name: str | None = None, allow_partial: bool = False)`
- Produces: `build_weighted_sampler(dataset: EEGWindowDataset, seed: int) -> torch.utils.data.WeightedRandomSampler`
- Produces: `validate_processed_dataset(processed_root: str | Path, allow_partial: bool = False) -> dict[str, object]`

- [ ] **Step 1: Add failing mmap dataset tests**

Create complete synthetic split metadata plus subject shards and a manifest that indexes their rows. Assert lazy mmap loading, exact normalization, metadata return, split and dataset filtering, and deterministic sampler weights. Create a separate `partial=true` fixture and assert default validation rejects it while `allow_partial=True` permits structural validation and `split=None` exposes every materialized row.

```python
eeg, label, metadata = dataset[0]
self.assertEqual(tuple(eeg.shape), (28, 2000))
self.assertEqual(eeg.dtype, torch.float32)
self.assertLess(float(torch.abs(eeg.mean(dim=-1)).max()), 1e-5)
self.assertEqual(label.dtype, torch.long)
self.assertIn("source_nback", metadata)
```

- [ ] **Step 2: Run dataset tests and verify failure**

Run: `python -m unittest tests.test_eeg_pretraining.DatasetTests -v`

Expected: FAIL because `EEGWindowDataset` is missing.

- [ ] **Step 3: Implement manifest validation, lazy memmaps, and normalization**

Cache open NumPy memmaps by absolute path inside each worker process. Normalize each channel with:

```python
mean = x.mean(axis=-1, keepdims=True)
std = x.std(axis=-1, keepdims=True)
x = (x - mean) / (std + 1e-6)
```

Return `(torch.Tensor, torch.tensor(label, dtype=torch.long), metadata dict)`. Validation must reject subject overlap, missing files, row-count mismatches, non-finite arrays, wrong shapes, or missing labels in any split/dataset pair. Unless `allow_partial=True`, reject `qc_summary.json` with `partial=true`; partial validation still enforces every structural check but skips complete subject-count and per-split class-coverage requirements. Training always uses the default `allow_partial=False`.

- [ ] **Step 4: Run dataset tests**

Run: `python -m unittest tests.test_eeg_pretraining.DatasetTests -v`

Expected: PASS.

- [ ] **Step 5: Commit mmap dataset support**

```bash
git add eeg_pretraining/data.py eeg_pretraining/__init__.py tests/test_eeg_pretraining.py
git commit -m "feat: load EEG pretraining shards with mmap"
```

---

### Task 7: EEG Model and Strict V14 Checkpoint Transfer

**Files:**
- Create: `eeg_pretraining/modeling.py`
- Modify: `eeg_pretraining/__init__.py`
- Modify: `tests/test_eeg_pretraining.py`

**Interfaces:**
- Produces: `EEGPretrainModel(eeg_encoder: nn.Module, eeg_adapter: nn.Module, eeg_aux_head: nn.Module)`
- Produces: `V14_SOURCE_CONFIG: dict[str, object]`
- Produces: `extract_eeg_state_dict(source_state: Mapping[str, torch.Tensor], expected_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]`
- Produces: `overlay_eeg_state_dict(v14_state: Mapping[str, torch.Tensor], eeg_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]`
- Produces: `build_eeg_pretrain_model(checkpoint_path: str | Path) -> EEGPretrainModel`

- [ ] **Step 1: Add failing model and checkpoint tests**

Test that the model uses the exact three top-level attributes, accepts `(2, 28, 2000)`, returns `(2, 3)`, and retains a 30-row channel embedding table.

Create a synthetic full state dictionary and a compact expected dictionary with EEG and non-EEG keys. Verify extraction rejects missing, unexpected, duplicate-after-container-unwrapping, and shape-mismatched EEG keys. Verify overlay changes every EEG tensor while preserving every non-EEG tensor with `torch.equal`.

- [ ] **Step 2: Run model tests and verify failure**

Run: `python -m unittest tests.test_eeg_pretraining.ModelAndCheckpointTests -v`

Expected: FAIL because `eeg_pretraining.modeling` does not exist.

- [ ] **Step 3: Implement model forward and strict transfer**

```python
class EEGPretrainModel(nn.Module):
    def __init__(self, eeg_encoder, eeg_adapter, eeg_aux_head):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.eeg_adapter = eeg_adapter
        self.eeg_aux_head = eeg_aux_head

    def forward(self, x_eeg):
        tokens = self.eeg_encoder.forward_tokens(x_eeg)
        features = self.eeg_adapter(tokens.mean(dim=1))
        return self.eeg_aux_head(features)
```

Define `V14_SOURCE_CONFIG` with the exact source construction values: EEG `emb_size=256`, `heads=8`, `depth=4`, `n_channels=30`, `n_fft=200`, `hop_length=100`; `nirs_channels=36`, `n_classes=3`, `fusion_dim=256`, `nirs_layout="block"`, `nirs_dim_model=64`, `nirs_depth=2`, `nirs_heads=4`, `nirs_dropout=0.2`, `classifier_dropout=0.3`, `preblock_residual_scale=0.1`, `preblock_mode="legacy"`, `cross_attn_dropout=0.1`, `cross_delay_frames=2`, `cross_window_frames=8`, `cross_alignment_mode="local"`, `cross_fusion_mode="logit"`, `aux_logit_scale_init=0.01`, `nirs_hb_mode="mixed_split_residual"`, `nirs_hb_branch_mode="adapter"`, `hb_split_scale_init=0.01`, and `hb_logit_scale_init=0.02`.

Load either a raw state dictionary or a structured checkpoint containing exactly `model_state_dict`. Instantiate `MultimodalBIOT_V14` from `V14_SOURCE_CONFIG`, load the full source checkpoint strictly, deep-copy only the three EEG modules, and return the compact model. Strip no prefixes by heuristic; accept only exact production prefixes. Extraction compares the source whitelist against `expected_state` for exact key set, shape, and dtype. Overlay requires exact key equality with the three whitelist prefixes present in the target V14 state and returns cloned tensors so the input mapping is not mutated.

- [ ] **Step 4: Run model and checkpoint tests**

Run: `python -m unittest tests.test_eeg_pretraining.ModelAndCheckpointTests -v`

Expected: PASS.

- [ ] **Step 5: Commit EEG model and transfer contract**

```bash
git add eeg_pretraining/modeling.py eeg_pretraining/__init__.py tests/test_eeg_pretraining.py
git commit -m "feat: add strict V14 EEG checkpoint transfer"
```

---

### Task 8: Staged Training and Metrics

**Files:**
- Create: `eeg_pretraining/training.py`
- Modify: `eeg_pretraining/__init__.py`
- Modify: `tests/test_eeg_pretraining.py`

**Interfaces:**
- Produces: `TrainingConfig(seed: int = 42, batch_size: int = 32, epochs: int = 50, freeze_encoder_epochs: int = 5, encoder_lr: float = 1e-5, head_lr: float = 1e-4, weight_decay: float = 1e-4, label_smoothing: float = 0.05, mixup_alpha: float = 0.2, early_stopping_patience: int = 10)`
- Produces: `set_encoder_trainable(model: EEGPretrainModel, trainable: bool) -> None`
- Produces: `build_optimizer(model: EEGPretrainModel, config: TrainingConfig) -> torch.optim.AdamW`
- Produces: `evaluate_by_dataset(model: EEGPretrainModel, loader: DataLoader, device: torch.device) -> dict[str, object]`
- Produces: `train_eeg_model(model: EEGPretrainModel, loaders: Mapping[str, DataLoader], config: TrainingConfig, output_dir: str | Path, metadata: Mapping[str, object]) -> dict[str, object]`

- [ ] **Step 1: Add failing staged-training and metric tests**

Use a tiny fake encoder/adapter/head model and a deterministic two-dataset loader. Assert that encoder parameters are frozen before epoch 5 and trainable afterward, optimizer groups use `1e-5` and `1e-4`, and checkpoint score equals the arithmetic mean of the two dataset macro-F1 values.

Add a one-batch CPU smoke test with mixup enabled and assert finite loss plus creation of `best.pth`, `last.pth`, and `history.json`.

- [ ] **Step 2: Run training tests and verify failure**

Run: `python -m unittest tests.test_eeg_pretraining.TrainingTests -v`

Expected: FAIL because `eeg_pretraining.training` does not exist.

- [ ] **Step 3: Implement deterministic staged training**

Use `torch.manual_seed`, NumPy seed, CUDA seeds, deterministic cuDNN, AdamW parameter groups, label-smoothed cross entropy, training-only mixup, `ReduceLROnPlateau(mode="max")`, and patience-10 early stopping. Keep the frozen encoder in evaluation mode for epochs 0-4, then restore training mode and gradients from epoch 5 onward.

Metrics must use scikit-learn with `labels=[0, 1, 2]` and `zero_division=0`. Scheduler updates, early stopping, and best-checkpoint selection use only the arithmetic mean of the two validation macro-F1 scores. Do not evaluate the test loaders during epoch selection; after training, reload `best.pth`, evaluate each test dataset once, and append those results to the checkpoint and `history.json`.

Save structured checkpoints with `format_version=1`, `model_state_dict`, source checkpoint, channel order, label mapping, split lists, config, epoch, and metrics. Create `output_dir` only after full-data validation and strict source-checkpoint compatibility checks succeed.

- [ ] **Step 4: Run training tests**

Run: `python -m unittest tests.test_eeg_pretraining.TrainingTests -v`

Expected: PASS.

- [ ] **Step 5: Commit staged training**

```bash
git add eeg_pretraining/training.py eeg_pretraining/__init__.py tests/test_eeg_pretraining.py
git commit -m "feat: train V14 EEG path across datasets"
```

---

### Task 9: CLIs, Documentation, and Repository Verification

**Files:**
- Create: `eeg_pretraining/cli.py`
- Create: `prepare_eeg_pretraining.py`
- Create: `run_eeg_pretraining.py`
- Modify: `README.md`
- Modify: `tests/test_eeg_pretraining.py`

**Interfaces:**
- Produces: `build_prepare_parser() -> argparse.ArgumentParser`
- Produces: `build_train_parser() -> argparse.ArgumentParser`
- Produces preprocessing flag: `--limit_subjects_per_dataset`
- Produces training flag: `--validate_only`

- [ ] **Step 1: Add failing parser and entry-point tests**

Assert defaults `seed=42`, `batch_size=32`, `epochs=50`, `freeze_encoder_epochs=5`, `encoder_lr=1e-5`, `head_lr=1e-4`, `mixup_alpha=0.2`, and strict required path arguments. Verify importing both root entry points has no side effects.

- [ ] **Step 2: Run CLI tests and verify failure**

Run: `python -m unittest tests.test_eeg_pretraining.CLITests -v`

Expected: FAIL because CLI modules and entry points are missing.

- [ ] **Step 3: Implement thin entry points and document commands**

`prepare_eeg_pretraining.py` must only parse arguments and call `prepare_eeg_data`. `run_eeg_pretraining.py` must inspect `qc_summary.json`, call `validate_processed_dataset(..., allow_partial=summary["partial"])` only in `--validate_only` mode, build the model from the source checkpoint, and run a forward/overlay compatibility check. Without `--validate_only`, it must reject partial data before creating loaders or an optimizer.

Add these exact workflow forms to `README.md`:

```bash
python prepare_eeg_pretraining.py \
  --cognitive_workload_root "/mnt/data/wrf/EEGfNIRS/datasets/Cognitive Workload" \
  --executive_functioning_root "/mnt/data/wrf/EEGfNIRS/datasets/Executive Functioning" \
  --output_dir /mnt/data/wrf/EEGfNIRS/processed/eeg_pretraining_v1_dry_run \
  --limit_subjects_per_dataset 1

python run_eeg_pretraining.py \
  --data_path /mnt/data/wrf/EEGfNIRS/processed/eeg_pretraining_v1_dry_run \
  --checkpoint_path /mnt/data/wrf/BIOT/reliable_subjectgap_seed42_v14_hbadapter_hblogit002.pth \
  --save_dir /mnt/data/wrf/EEGfNIRS/runs/eeg_pretraining_v1_dry_run_validation \
  --validate_only
```

- [ ] **Step 4: Run the complete repository test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all existing and new tests PASS with zero failures and zero errors.

- [ ] **Step 5: Inspect code quality and commit**

Run: `git diff --check`

Expected: no whitespace errors.

```bash
git add eeg_pretraining/cli.py prepare_eeg_pretraining.py run_eeg_pretraining.py README.md tests/test_eeg_pretraining.py
git commit -m "docs: add EEG pretraining workflow"
```

---

### Task 10: Real One-Subject Dry Runs and Checkpoint Integration Gate

**Files:**
- No source edits expected
- Generate outside repository: `/mnt/data/wrf/EEGfNIRS/processed/eeg_pretraining_v1_dry_run`

**Interfaces:**
- Consumes: preprocessing and validation CLIs from Task 9
- Produces: dry-run `manifest.jsonl`, `splits.json`, `qc_summary.json`, and strict real-checkpoint validation output

- [ ] **Step 1: Run one subject from each source dataset**

Run the documented preprocessing command with `--limit_subjects_per_dataset 1`.

Expected: exit 0, one Cognitive Workload subject directory, one Executive Functioning subject directory, and no partial temporary directories.

- [ ] **Step 2: Audit dry-run arrays and metadata**

Run: `python -c "import json,numpy as np,pathlib; p=pathlib.Path('/mnt/data/wrf/EEGfNIRS/processed/eeg_pretraining_v1_dry_run'); rows=[json.loads(x) for x in (p/'manifest.jsonl').read_text().splitlines()]; arrays={x:np.load(p/x,mmap_mode='r') for x in {r['x_path'] for r in rows}}; assert all(a.ndim==3 and a.shape[1:]==(28,2000) and np.isfinite(a).all() for a in arrays.values()); assert all(0<=r['row_index']<arrays[r['x_path']].shape[0] for r in rows); print(len(rows)); print(sorted({(r['dataset'],r['mapped_label']) for r in rows})); print(json.loads((p/'qc_summary.json').read_text()))"`

Expected: every referenced array exists, every array row is `(28, 2000)`, all values are finite, and QC output reports accepted/rejected counts rather than silent skips.

- [ ] **Step 3: Validate the real V14 checkpoint**

Run the documented training command with `--validate_only`; the command automatically permits partial structural validation but never constructs training loaders.

Expected: the raw 299-tensor source checkpoint loads strictly, extraction reports 67 EEG tensors, the EEG model accepts a dry-run batch, and the overlay check reports zero changes among the 232 non-EEG tensors.

- [ ] **Step 4: Re-run all unit tests after real integration**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 5: Stop for dry-run review**

Report selected subject IDs, accepted/rejected windows by class and dataset, interpolated channels, checkpoint key counts, and any warnings. Do not start full preprocessing until this report is accepted.

---

### Task 11: Full EEG Preprocessing and QC Gate

**Files:**
- No source edits expected
- Generate outside repository: `/mnt/data/wrf/EEGfNIRS/processed/eeg_pretraining_v1`

**Interfaces:**
- Consumes: approved dry-run pipeline
- Produces: complete processed EEG dataset and QC report

- [ ] **Step 1: Run full preprocessing without the subject limit**

```bash
python prepare_eeg_pretraining.py \
  --cognitive_workload_root "/mnt/data/wrf/EEGfNIRS/datasets/Cognitive Workload" \
  --executive_functioning_root "/mnt/data/wrf/EEGfNIRS/datasets/Executive Functioning" \
  --output_dir /mnt/data/wrf/EEGfNIRS/processed/eeg_pretraining_v1 \
  --seed 42
```

Expected: exit 0 with 18 Cognitive Workload and 24 Executive Functioning subjects represented.

- [ ] **Step 2: Validate processed dataset contracts**

Run the training entry point with `--validate_only` against the full output.

Expected: split disjointness, all class coverage, finite `(28, 2000)` arrays, strict checkpoint load, and non-EEG preservation all pass.

- [ ] **Step 3: Audit class, subject, and rejection counts**

Inspect `qc_summary.json` and `splits.json`. Confirm exact 12/3/3 and 16/4/4 subject counts, no missing classes in any dataset/split, and no unexplained subject failures.

- [ ] **Step 4: Check disk and manifest consistency**

Run: `python -c "from eeg_pretraining.data import validate_processed_dataset; import json; print(json.dumps(validate_processed_dataset('/mnt/data/wrf/EEGfNIRS/processed/eeg_pretraining_v1'), indent=2))"`

Expected: reported sample count equals manifest rows and every shard row count.

- [ ] **Step 5: Stop for full QC review**

Report all QC counts and disk usage. Do not start the long-running training command until this report is accepted.

---

### Task 12: Full EEG Continued Pretraining and Final Handoff

**Files:**
- No source edits expected unless verification exposes a reproducible defect; any defect returns to a failing test before a code fix
- Generate outside repository: `/mnt/data/wrf/EEGfNIRS/runs/eeg_pretraining_v1`

**Interfaces:**
- Consumes: approved full processed dataset and known V14 checkpoint
- Produces: `best.pth`, `last.pth`, `history.json`, per-dataset test metrics, and V14 overlay verification

- [ ] **Step 1: Start full training**

```bash
python run_eeg_pretraining.py \
  --data_path /mnt/data/wrf/EEGfNIRS/processed/eeg_pretraining_v1 \
  --checkpoint_path /mnt/data/wrf/BIOT/reliable_subjectgap_seed42_v14_hbadapter_hblogit002.pth \
  --save_dir /mnt/data/wrf/EEGfNIRS/runs/eeg_pretraining_v1 \
  --seed 42 \
  --batch_size 32 \
  --epochs 50 \
  --freeze_encoder_epochs 5 \
  --encoder_lr 1e-5 \
  --head_lr 1e-4 \
  --mixup_alpha 0.2
```

Expected: training completes or stops early after patience 10, and both best and final checkpoints are saved.

- [ ] **Step 2: Verify final artifacts and metrics**

Confirm `best.pth`, `last.pth`, and `history.json` are readable. Report best epoch, per-dataset validation/test accuracy and macro-F1, pooled metrics, and confusion matrices.

- [ ] **Step 3: Run final strict overlay verification**

Load the original full V14 checkpoint, overlay `best.pth`, verify all EEG keys match the pretraining checkpoint, and verify all non-EEG tensors equal the original values.

- [ ] **Step 4: Run the complete test suite one final time**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS with zero failures and zero errors.

- [ ] **Step 5: Record the handoff**

Report artifact paths, exact command, seed, source checkpoint, subject splits, QC summary, best metric, test metrics, and the known target-checkpoint selection limitation. Stop before fNIRS or fusion implementation.
