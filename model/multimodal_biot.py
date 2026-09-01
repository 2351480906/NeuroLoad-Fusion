import torch
import torch.nn as nn
from model.biot import BIOTEncoder
import torch.nn.functional as F

# ==========================================
# 1. fNIRS 编码器 (ComplexNIRS_Encoder)
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResBlock1D, self).__init__()
        padding = (kernel_size - 1) 
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ComplexNIRS_Encoder(nn.Module):
    def __init__(self, in_channels, out_dim=128):
        super().__init__()
        # 简化版：去掉 GRU，减少层数，保留多尺度卷积和 SEBlock        
        # 1. 多尺度输入 (捕捉不同快慢的血氧变化)
        self.branch1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn_concat = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()        
        # 2. 浅层特征提取
        self.conv1 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2) # 100 -> 50        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2) # 50 -> 25     
        # 3. 空间注意力 (保留 SEBlock，这很有用)
        self.se = SEBlock(256)        
        # 4. 全局池化代替 GRU
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.5) # 增加 Dropout 防止过拟合
        )

    def _forward_conv_features(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.relu(self.bn_concat(x))
        
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        x = self.se(x)
        return x

    def forward(self, x):
        # x: (B, C, 100)
        x = self._forward_conv_features(x)
        
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def forward_tokens(self, x):
        # x: (B, C, T) -> (B, T/4, out_dim)
        x = self._forward_conv_features(x)
        x = x.transpose(1, 2)
        return self.fc(x)

    def forward_features_and_tokens(self, x):
        # Run the convolutional trunk once and derive both pooled and temporal views.
        conv_features = self._forward_conv_features(x)
        pooled = self.global_pool(conv_features).squeeze(-1)
        feat = self.fc(pooled)
        tokens = self.fc(conv_features.transpose(1, 2))
        return feat, tokens

# ==========================================
# 1. 更复杂的fNIRS_Encoder
# ==========================================
class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 这里的卷积核大小为 1，作用于时间维度上的每个点
        self.conv1 = nn.Conv1d(channels, channels // 8, kernel_size=1) 
        self.conv2 = nn.Conv1d(channels // 8, 1, kernel_size=1) # 输出 1 个通道的 mask

    def forward(self, x):
        # x: (B, C, T)
        mask = torch.tanh(self.conv1(x))
        mask = torch.sigmoid(self.conv2(mask)) # (B, 1, T) 时间注意力图
        return x * mask


class AdvancedNIRS_Encoder(nn.Module):
    def __init__(self, in_channels, out_dim=128):
        super().__init__()
        
        # 1. 增强的多尺度输入
        self.branch1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.branch3 = nn.Conv1d(in_channels, 32, kernel_size=11, padding=5) # 增加更大感受野
        self.bn_concat = nn.BatchNorm1d(96) # 32*3
        
        # 2. 深度特征提取
        self.main_conv = nn.Sequential(
            nn.Conv1d(96, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 3. 双重注意力
        self.channel_att = SEBlock(256)      # 关注哪些通道重要
        self.temporal_att = TemporalAttention(256) # 关注哪个时刻重要
        
        # 4. 输出
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # 多尺度融合
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.bn_concat(x))
        
        # 主干提取
        x = self.main_conv(x)
        
        # 注意力机制
        x = self.channel_att(x)
        x = self.temporal_att(x)
        
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x


# ==========================================
# 2. 动态 fNIRS 2D-Transformer 编码器
# ==========================================
class AdaptiveFNIRSPreBlock(nn.Module):
    """Residual adaptive filtering over the fNIRS time axis.

    The block estimates sample-wise low-frequency trends with multi-scale
    temporal averages and subtracts a learned, gated trend. The residual path
    keeps slow task-related hemodynamic responses available to the model.
    """

    def __init__(self, hidden_channels=16, residual_scale=0.1, mode="legacy"):
        super().__init__()
        if mode not in {"legacy", "weighted"}:
            raise ValueError("PreBlock mode must be either 'legacy' or 'weighted'.")
        self.mode = mode
        self.pools = nn.ModuleList([
            nn.AvgPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1), count_include_pad=False),
            nn.AvgPool2d(kernel_size=(1, 7), stride=1, padding=(0, 3), count_include_pad=False),
            nn.AvgPool2d(kernel_size=(1, 15), stride=1, padding=(0, 7), count_include_pad=False),
            nn.AvgPool2d(kernel_size=(1, 31), stride=1, padding=(0, 15), count_include_pad=False),
        ])
        self.filter_net = nn.Sequential(
            nn.Conv2d(14, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 4, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.scale_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(14, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 8, kernel_size=1),
        )
        self.strength_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(14, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )
        nn.init.zeros_(self.scale_net[-1].weight)
        nn.init.zeros_(self.scale_net[-1].bias)
        nn.init.zeros_(self.strength_net[-1].weight)
        nn.init.constant_(self.strength_net[-1].bias, -2.1972246)

    def forward(self, x):
        # x: (B, 2, C, T)
        trends = [pool(x) for pool in self.pools]
        features = torch.cat([x, *trends, x - trends[1], x - trends[-1]], dim=1)
        if self.mode == "weighted":
            batch = x.size(0)
            scale_weights = self.scale_net(features).view(batch, 2, len(trends), 1, 1)
            scale_weights = scale_weights.softmax(dim=2)
            stacked_trends = torch.stack(trends, dim=2)
            trend = (scale_weights * stacked_trends).sum(dim=2)
            strength = torch.sigmoid(self.strength_net(features))
            filtered = x - strength * trend
            normalized = F.layer_norm(filtered, filtered.shape[-1:])
            return normalized + self.residual_scale * x

        trend, gate = self.filter_net(features).chunk(2, dim=1)
        filtered = x - torch.sigmoid(gate) * trend
        normalized = F.layer_norm(filtered, filtered.shape[-2:])
        return normalized + self.residual_scale * x


class AxialFNIRSTransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, ff_mult=4):
        super().__init__()
        self.norm_spatial = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_temporal = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, D, C, T)
        b, d, c, t = x.shape

        spatial = x.permute(0, 3, 2, 1).reshape(b * t, c, d)
        spatial_norm = self.norm_spatial(spatial)
        spatial_out, _ = self.spatial_attn(spatial_norm, spatial_norm, spatial_norm, need_weights=False)
        spatial = spatial + self.dropout(spatial_out)
        x = spatial.reshape(b, t, c, d).permute(0, 3, 2, 1)

        temporal = x.permute(0, 2, 3, 1).reshape(b * c, t, d)
        temporal_norm = self.norm_temporal(temporal)
        temporal_out, _ = self.temporal_attn(temporal_norm, temporal_norm, temporal_norm, need_weights=False)
        temporal = temporal + self.dropout(temporal_out)

        ff_in = self.norm_ff(temporal)
        temporal = temporal + self.dropout(self.ff(ff_in))
        x = temporal.reshape(b, c, t, d).permute(0, 3, 1, 2)
        return x


class DynamicFNIRS2DTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_dim=128,
        layout="block",
        dim=64,
        depth=2,
        heads=4,
        dropout=0.2,
    ):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(f"fNIRS input channels must be even for HbO/HbR pairing, got {in_channels}.")
        if layout not in {"block", "interleave"}:
            raise ValueError("layout must be either 'block' or 'interleave'.")

        self.in_channels = in_channels
        self.layout = layout
        self.pre = AdaptiveFNIRSPreBlock()
        self.chrom_gate = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
            nn.Linear(8, 2),
            nn.Sigmoid(),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(2, dim, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.GroupNorm(1, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            AxialFNIRSTransformerBlock(dim=dim, heads=heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.attn_pool = nn.Conv2d(dim, 1, kernel_size=1)
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def _reshape_chromophores(self, x):
        # x: (B, C_total, T) -> (B, 2, C_nirs, T)
        if x.size(1) % 2 != 0:
            raise ValueError(f"fNIRS batch has odd channel count: {x.size(1)}")
        half = x.size(1) // 2
        if self.layout == "block":
            return torch.stack([x[:, :half], x[:, half:]], dim=1)
        return torch.stack([x[:, 0::2], x[:, 1::2]], dim=1)

    def forward(self, x):
        x = self._reshape_chromophores(x)
        x = self.pre(x)

        gate = self.chrom_gate(x.mean(dim=(2, 3))).view(x.size(0), 2, 1, 1)
        x = x * gate

        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        attn = self.attn_pool(x).flatten(2).softmax(dim=-1)
        attn_pool = (x.flatten(2) * attn).sum(dim=-1)
        return self.proj(torch.cat([avg_pool, max_pool, attn_pool], dim=1))

# ==========================================
# 3. 多模态融合模型 (MultimodalBIOT)
# ==========================================
class MultimodalBIOT(nn.Module):
    def __init__(self, eeg_args, nirs_channels, n_classes, fusion_dim=256):
        super().__init__()
        
        # EEG 分支
        self.eeg_encoder = BIOTEncoder(**eeg_args)
        # 假设 BIOT 的 emb_size 是特征维度
        self.eeg_dim = eeg_args['emb_size'] 
        
        # NIRS 分支
        self.nirs_dim = 128
        self.nirs_encoder = AdvancedNIRS_Encoder(in_channels=nirs_channels, out_dim=self.nirs_dim)
        
        # 融合层 (Gated Fusion)
        total_dim = self.eeg_dim + self.nirs_dim
        self.fusion_gate = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, n_classes)
        )

    def forward(self, x_eeg, x_nirs):
        # 提取特征
        # 注意: 确保在 model/biot.py 中实现了 forward_features
        feat_eeg = self.eeg_encoder.forward_features(x_eeg) 
        feat_nirs = self.nirs_encoder(x_nirs)
        
        # === 新增：Modality Dropout (仅在训练时生效) ===
        if self.training:
            # 15% 的概率只用 fNIRS，15% 的概率只用 EEG
            p = torch.rand(1).item()
            if p < 0.15:
                feat_eeg = torch.zeros_like(feat_eeg)
            elif p < 0.30:
                feat_nirs = torch.zeros_like(feat_nirs)
        
        # 拼接与门控融合
        combined = torch.cat([feat_eeg, feat_nirs], dim=1)
        gate = self.fusion_gate(combined)
        combined = combined * gate
        
        logits = self.classifier(combined)
        return logits
