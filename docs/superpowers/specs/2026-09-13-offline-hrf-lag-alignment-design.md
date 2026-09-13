# Offline HRF-Prior Lag Alignment

## Goal

Replace V14's fixed one-second EEG-to-fNIRS delay mask with an offline,
time-coordinate-aware soft lag prior. The new path must preserve the existing
hard-mask implementation as an ablation baseline and must not interpret an
unavailable pre-window EEG frame as valid history.

The initial target is offline 10-second fNIRS-window classification on
Shin2018. It is deliberately not a real-time implementation: the sample can
include EEG history before the fNIRS query window. A later causal deployment
path can reuse the same lag representation with a ring buffer.

## Data Contract

Each offline sample has a query interval `[t, t + 10 s]`:

```text
X_nirs_query: (36, 100), sampled at 10 Hz over [t, t + 10 s]
X_eeg_context: (30, 3600), sampled at 200 Hz over [t - 8 s, t + 10 s]
```

The dataset loader must expose the actual start times for both tensors or,
when materialized windows have the fixed contract above, construct their time
coordinates deterministically. The EEG STFT frame centre is
`(frame * hop_length + n_fft / 2) / eeg_fs` relative to the EEG context
start. fNIRS temporal token centres are nominally derived from the fourfold
pooling stride. Boundary padding or missing history is represented by a valid
mask, never by a circular shift or a fallback key.

The legacy `(30, 2000)` EEG input remains available only in legacy alignment
mode. The HRF-prior alignment mode requires the extended EEG context.

## EEG Token Contract

The delay-aware layer must receive EEG tokens indexed by `(time, channel)`,
with an explicit timestamp per time frame. Temporal self-attention must not
let an EEG value token encode later EEG frames before lag attention applies.
The offline path may use a separate non-causal EEG representation for the
unimodal classifier, but the key/value sequence used by HRF lag alignment must
be time-safe.

## HRF-Prior Soft Lag Attention

For fNIRS query token `i` at time `t_n[i]` and EEG key `j` at time `t_e[j]`,
define `delta[i, j] = t_n[i] - t_e[j]`. Keys with `delta < 0` or
`delta > lag_max_sec` are invalid.

Valid attention logits receive an additive lag bias:

```text
score[i, j] = Q[i] K[j] / sqrt(d) + log w_branch(bin(delta[i, j]))
```

`w_branch` is a non-negative, normalized distribution on 0.5-second bins from
0 to 8 seconds. It is initialized from a broad Gaussian prior centred at 5
seconds and is learned separately for mixed, HbO, and HbR branches. A smooth
total-variation penalty is applied to neighboring bin probabilities. The
initial scope shares the distribution across channels and query times to keep
the parameter count appropriate for the available participant count.

The lag bias supplements content attention; it does not turn an HRF prior into
a claim of per-token causal identification. The learned lag distributions must
be logged for every validation/test run.

## Compatibility and Configuration

`cross_alignment_mode` gains `"soft_hrf"`; existing `"local"` and `"global"`
behavior is preserved for historical checkpoints. New arguments are:

```text
--eeg_sample_rate 200
--nirs_sample_rate 10
--offline_eeg_history_sec 8.0
--hrf_lag_max_sec 8.0
--hrf_lag_bin_sec 0.5
--hrf_prior_mean_sec 5.0
--hrf_prior_std_sec 1.5
--hrf_lag_smoothness_weight 0.01
```

The model rejects incompatible input lengths in `soft_hrf` mode rather than
silently reshaping a legacy 10-second EEG window.

## Evaluation

All lag selection and regularizer tuning use subject-independent validation.
The held-out test set is evaluated once for the chosen configuration. Required
ablations are:

1. Legacy fixed 1-second local mask.
2. Fixed-lag masks over 0, 2, 4, 5, 6, 7, and 8 seconds.
3. Soft HRF-prior lag attention.
4. Soft HRF-prior attention with paired EEG/fNIRS samples shuffled.

Report macro F1 and subject-level confidence intervals, the learned lag
distribution, and the difference from the strongest fixed-lag baseline.

## Tests

Unit tests must cover timestamp construction, allowed lag boundaries,
normalization and initial shape of the lag distribution, smoothness penalty,
invalid all-key rows, legacy-mode compatibility, and rejection of insufficient
EEG context. An integration smoke test must run both legacy and `soft_hrf`
forwards with their respective tensor contracts.
