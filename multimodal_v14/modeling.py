import torch
import torch.nn as nn

from model.biot import BIOTEncoder
from model.multimodal_biot import AdaptiveFNIRSPreBlock, ComplexNIRS_Encoder


class FeatureAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

    def forward(self, x):
        return x + self.net(x)


class TokenLevelDelayAwareFNIRSToEEGCrossAttention(nn.Module):
    def __init__(self, nirs_dim, eeg_dim, n_eeg_channels, heads=8, dropout=0.1,
                 delay_frames=2, window_frames=8, alignment_mode="local"):
        super().__init__()
        if alignment_mode not in {"global", "local"}:
            raise ValueError("alignment_mode must be either 'global' or 'local'.")
        self.n_eeg_channels = n_eeg_channels
        self.delay_frames = delay_frames
        self.window_frames = window_frames
        self.alignment_mode = alignment_mode
        self.q_norm = nn.LayerNorm(nirs_dim)
        self.kv_norm = nn.LayerNorm(eeg_dim)
        self.q_proj = nn.Linear(nirs_dim, eeg_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=eeg_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(eeg_dim, nirs_dim),
            nn.LayerNorm(nirs_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.ctx_gate = nn.Sequential(
            nn.Linear(nirs_dim * 2, nirs_dim),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.token_pool = nn.Linear(nirs_dim, 1)

    def forward(self, nirs_tokens, eeg_tokens):
        query = self.q_proj(self.q_norm(nirs_tokens))
        eeg_tokens = self.kv_norm(eeg_tokens)
        attn_mask = None
        if self.alignment_mode == "global":
            eeg_tokens = self._select_delayed_tokens(eeg_tokens)
        else:
            attn_mask = self._build_local_delay_mask(
                nirs_len=nirs_tokens.size(1),
                eeg_seq_len=eeg_tokens.size(1),
                device=eeg_tokens.device,
            )
        context, _ = self.attn(query, eeg_tokens, eeg_tokens, attn_mask=attn_mask, need_weights=False)
        context = self.ctx_proj(context)
        gate = self.ctx_gate(torch.cat([nirs_tokens, context], dim=-1))
        context = gate * context
        weights = self.token_pool(nirs_tokens).softmax(dim=1)
        context = (context * weights).sum(dim=1)
        return self.alpha * context

    def _select_delayed_tokens(self, eeg_tokens):
        batch, seq_len, dim = eeg_tokens.shape
        if seq_len % self.n_eeg_channels != 0:
            return eeg_tokens

        frames = seq_len // self.n_eeg_channels
        tokens = eeg_tokens.view(batch, self.n_eeg_channels, frames, dim)
        end = max(frames - self.delay_frames, 1)
        start = max(end - self.window_frames, 0)
        selected = tokens[:, :, start:end, :]
        return selected.reshape(batch, -1, dim)

    def _build_local_delay_mask(self, nirs_len, eeg_seq_len, device):
        if eeg_seq_len % self.n_eeg_channels != 0:
            return None

        frames = eeg_seq_len // self.n_eeg_channels
        mask = torch.ones(nirs_len, self.n_eeg_channels, frames, dtype=torch.bool, device=device)
        for idx in range(nirs_len):
            anchor = ((idx + 1) * frames + nirs_len - 1) // nirs_len
            end = min(max(anchor - self.delay_frames, 1), frames)
            start = max(end - self.window_frames, 0)
            mask[idx, :, start:end] = False
        return mask.view(nirs_len, eeg_seq_len)


class MultimodalBIOT_V14(nn.Module):
    def __init__(self, eeg_args, nirs_channels, n_classes, fusion_dim=256,
                 nirs_layout="block", nirs_dim_model=64, nirs_depth=2,
                 nirs_heads=4, nirs_dropout=0.2, classifier_dropout=0.3,
                 preblock_residual_scale=0.1, preblock_mode="legacy", cross_attn_dropout=0.1,
                 cross_delay_frames=2, cross_window_frames=8,
                 cross_alignment_mode="local", cross_fusion_mode="logit", aux_logit_scale_init=0.01,
                 nirs_hb_mode="mixed", nirs_hb_branch_mode="shared",
                 hb_split_scale_init=0.01, hb_logit_scale_init=0.0):
        super().__init__()
        if nirs_channels % 2 != 0:
            raise ValueError(f"fNIRS input channels must be even for HbO/HbR pairing, got {nirs_channels}.")
        if nirs_layout not in {"block", "interleave"}:
            raise ValueError("nirs_layout must be either 'block' or 'interleave'.")
        if cross_fusion_mode not in {"residual", "concat", "logit"}:
            raise ValueError("cross_fusion_mode must be one of 'residual', 'concat', or 'logit'.")
        if nirs_hb_mode not in {"mixed", "split_shared", "mixed_split_residual"}:
            raise ValueError("nirs_hb_mode must be 'mixed', 'split_shared', or 'mixed_split_residual'.")
        if nirs_hb_branch_mode not in {"shared", "separate", "adapter"}:
            raise ValueError("nirs_hb_branch_mode must be 'shared', 'separate', or 'adapter'.")
        self.eeg_encoder = BIOTEncoder(**eeg_args)
        self.eeg_dim = eeg_args['emb_size']
        self.eeg_adapter = FeatureAdapter(self.eeg_dim)
        self.nirs_dim = 128
        self.nirs_layout = nirs_layout
        self.cross_fusion_mode = cross_fusion_mode
        self.nirs_hb_mode = nirs_hb_mode
        self.nirs_hb_branch_mode = nirs_hb_branch_mode
        self.nirs_preblock = AdaptiveFNIRSPreBlock(
            residual_scale=preblock_residual_scale,
            mode=preblock_mode,
        )
        self.nirs_encoder = ComplexNIRS_Encoder(in_channels=nirs_channels, out_dim=self.nirs_dim)
        self.nirs_hb_encoder = ComplexNIRS_Encoder(in_channels=nirs_channels // 2, out_dim=self.nirs_dim)
        self.nirs_hbo_encoder = ComplexNIRS_Encoder(in_channels=nirs_channels // 2, out_dim=self.nirs_dim)
        self.nirs_hbr_encoder = ComplexNIRS_Encoder(in_channels=nirs_channels // 2, out_dim=self.nirs_dim)
        self.hbo_feature_adapter = FeatureAdapter(self.nirs_dim)
        self.hbr_feature_adapter = FeatureAdapter(self.nirs_dim)
        self.cross_fusion = TokenLevelDelayAwareFNIRSToEEGCrossAttention(
            nirs_dim=self.nirs_dim,
            eeg_dim=self.eeg_dim,
            n_eeg_channels=eeg_args['n_channels'],
            heads=nirs_heads,
            dropout=cross_attn_dropout,
            delay_frames=cross_delay_frames,
            window_frames=cross_window_frames,
            alignment_mode=cross_alignment_mode,
        )
        self.cross_hbo = TokenLevelDelayAwareFNIRSToEEGCrossAttention(
            nirs_dim=self.nirs_dim,
            eeg_dim=self.eeg_dim,
            n_eeg_channels=eeg_args['n_channels'],
            heads=nirs_heads,
            dropout=cross_attn_dropout,
            delay_frames=cross_delay_frames,
            window_frames=cross_window_frames,
            alignment_mode=cross_alignment_mode,
        )
        self.cross_hbr = TokenLevelDelayAwareFNIRSToEEGCrossAttention(
            nirs_dim=self.nirs_dim,
            eeg_dim=self.eeg_dim,
            n_eeg_channels=eeg_args['n_channels'],
            heads=nirs_heads,
            dropout=cross_attn_dropout,
            delay_frames=cross_delay_frames,
            window_frames=cross_window_frames,
            alignment_mode=cross_alignment_mode,
        )
        self.nirs_hb_gate = nn.Sequential(
            nn.Linear(self.nirs_dim * 2, self.nirs_dim),
            nn.Sigmoid(),
        )
        self.cross_hb_gate = nn.Sequential(
            nn.Linear(self.nirs_dim * 2, self.nirs_dim),
            nn.Sigmoid(),
        )
        self.hb_split_scale = nn.Parameter(torch.tensor(float(hb_split_scale_init)))

        total_dim = self.eeg_dim + self.nirs_dim
        self.cross_to_fusion = nn.Linear(self.nirs_dim, total_dim, bias=False)
        if self.cross_fusion_mode == "concat":
            total_dim = total_dim + self.nirs_dim
        self.fusion_gate = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.SiLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_dim, n_classes),
        )
        self.eeg_aux_head = nn.Sequential(
            nn.Linear(self.eeg_dim, fusion_dim // 2),
            nn.SiLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_dim // 2, n_classes),
        )
        self.nirs_aux_head = nn.Sequential(
            nn.Linear(self.nirs_dim, fusion_dim // 2),
            nn.SiLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_dim // 2, n_classes),
        )
        self.cross_logit_head = nn.Sequential(
            nn.Linear(self.nirs_dim, fusion_dim // 2),
            nn.SiLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_dim // 2, n_classes),
        )
        self.hbo_cross_head = nn.Sequential(
            nn.Linear(self.nirs_dim, fusion_dim // 2),
            nn.SiLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_dim // 2, n_classes),
        )
        self.hbr_cross_head = nn.Sequential(
            nn.Linear(self.nirs_dim, fusion_dim // 2),
            nn.SiLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_dim // 2, n_classes),
        )
        self.hb_logit_gate = nn.Sequential(
            nn.Linear(self.eeg_dim + self.nirs_dim * 3, n_classes),
            nn.Sigmoid(),
        )
        self.logit_gate = nn.Sequential(
            nn.Linear(self.eeg_dim + self.nirs_dim * 2, n_classes),
            nn.Sigmoid(),
        )
        self.aux_logit_scale = nn.Parameter(torch.tensor(float(aux_logit_scale_init)))
        self.hb_logit_scale = nn.Parameter(torch.tensor(float(hb_logit_scale_init)))

    def forward(self, x_eeg, x_nirs, return_aux=False):
        eeg_tokens = self.eeg_encoder.forward_tokens(x_eeg)
        feat_eeg = eeg_tokens.mean(dim=1)
        feat_eeg = self.eeg_adapter(feat_eeg)
        x_nirs_4d = self._apply_nirs_preblock_4d(x_nirs)
        context_hbo = None
        context_hbr = None
        if self.nirs_hb_mode == "split_shared":
            hbo = x_nirs_4d[:, 0]
            hbr = x_nirs_4d[:, 1]
            feat_hbo, hbo_tokens = self._encode_hbo(hbo)
            feat_hbr, hbr_tokens = self._encode_hbr(hbr)
            nirs_gate = self.nirs_hb_gate(torch.cat([feat_hbo, feat_hbr], dim=1))
            feat_nirs = nirs_gate * feat_hbo + (1.0 - nirs_gate) * feat_hbr
            context_hbo = self.cross_hbo(hbo_tokens, eeg_tokens)
            context_hbr = self.cross_hbr(hbr_tokens, eeg_tokens)
            cross_gate = self.cross_hb_gate(torch.cat([context_hbo, context_hbr], dim=1))
            cross_context = cross_gate * context_hbo + (1.0 - cross_gate) * context_hbr
        elif self.nirs_hb_mode == "mixed_split_residual":
            x_nirs = self._flatten_nirs_4d(x_nirs_4d)
            feat_nirs, nirs_tokens = self.nirs_encoder.forward_features_and_tokens(x_nirs)
            cross_context = self.cross_fusion(nirs_tokens, eeg_tokens)
            hbo = x_nirs_4d[:, 0]
            hbr = x_nirs_4d[:, 1]
            _, hbo_tokens = self._encode_hbo(hbo)
            _, hbr_tokens = self._encode_hbr(hbr)
            context_hbo = self.cross_hbo(hbo_tokens, eeg_tokens)
            context_hbr = self.cross_hbr(hbr_tokens, eeg_tokens)
            cross_gate = self.cross_hb_gate(torch.cat([context_hbo, context_hbr], dim=1))
            split_context = cross_gate * context_hbo + (1.0 - cross_gate) * context_hbr
            cross_context = cross_context + self.hb_split_scale * split_context
        else:
            x_nirs = self._flatten_nirs_4d(x_nirs_4d)
            feat_nirs, nirs_tokens = self.nirs_encoder.forward_features_and_tokens(x_nirs)
            cross_context = self.cross_fusion(nirs_tokens, eeg_tokens)
        combined = torch.cat([feat_eeg, feat_nirs], dim=1)
        if self.cross_fusion_mode == "concat":
            combined = torch.cat([combined, cross_context], dim=1)
        elif self.cross_fusion_mode == "residual":
            combined = combined + self.cross_to_fusion(cross_context)
        gate = self.fusion_gate(combined)
        combined = combined * gate
        logits = self.classifier(combined)
        if self.cross_fusion_mode == "logit":
            aux_logits = self.cross_logit_head(cross_context)
            logit_gate = self.logit_gate(torch.cat([feat_eeg, feat_nirs, cross_context], dim=1))
            logits = logits + self.aux_logit_scale * logit_gate * aux_logits
            if context_hbo is not None:
                logits_hbo = self.hbo_cross_head(context_hbo)
                logits_hbr = self.hbr_cross_head(context_hbr)
                hb_logit_gate = self.hb_logit_gate(torch.cat([feat_eeg, feat_nirs, context_hbo, context_hbr], dim=1))
                hb_logits = hb_logit_gate * logits_hbo + (1.0 - hb_logit_gate) * logits_hbr
                logits = logits + self.hb_logit_scale * hb_logits
        if return_aux:
            if context_hbo is None:
                logits_hbo = logits.new_zeros(logits.shape)
                logits_hbr = logits.new_zeros(logits.shape)
            else:
                logits_hbo = self.hbo_cross_head(context_hbo)
                logits_hbr = self.hbr_cross_head(context_hbr)
            return logits, self.eeg_aux_head(feat_eeg), self.nirs_aux_head(feat_nirs), logits_hbo, logits_hbr
        return logits

    def _encode_hbo(self, hbo):
        if self.nirs_hb_branch_mode == "separate":
            return self.nirs_hbo_encoder.forward_features_and_tokens(hbo)
        feat, tokens = self.nirs_hb_encoder.forward_features_and_tokens(hbo)
        if self.nirs_hb_branch_mode == "adapter":
            feat = self.hbo_feature_adapter(feat)
            tokens = self.hbo_feature_adapter(tokens)
        return feat, tokens

    def _encode_hbr(self, hbr):
        if self.nirs_hb_branch_mode == "separate":
            return self.nirs_hbr_encoder.forward_features_and_tokens(hbr)
        feat, tokens = self.nirs_hb_encoder.forward_features_and_tokens(hbr)
        if self.nirs_hb_branch_mode == "adapter":
            feat = self.hbr_feature_adapter(feat)
            tokens = self.hbr_feature_adapter(tokens)
        return feat, tokens

    def _apply_nirs_preblock(self, x_nirs):
        return self._flatten_nirs_4d(self._apply_nirs_preblock_4d(x_nirs))

    def _apply_nirs_preblock_4d(self, x_nirs):
        half = x_nirs.size(1) // 2
        if self.nirs_layout == "block":
            x_4d = torch.stack([x_nirs[:, :half], x_nirs[:, half:]], dim=1)
            return self.nirs_preblock(x_4d)
        x_4d = torch.stack([x_nirs[:, 0::2], x_nirs[:, 1::2]], dim=1)
        return self.nirs_preblock(x_4d)

    def _flatten_nirs_4d(self, x_4d):
        if self.nirs_layout == "block":
            return torch.cat([x_4d[:, 0], x_4d[:, 1]], dim=1)
        batch, _, channels, time_steps = x_4d.shape
        out = x_4d.new_empty(batch, channels * 2, time_steps)
        out[:, 0::2] = x_4d[:, 0]
        out[:, 1::2] = x_4d[:, 1]
        return out
