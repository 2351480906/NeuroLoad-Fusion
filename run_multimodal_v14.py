"""Backward-compatible entry point for the V14 multimodal training pipeline."""

from multimodal_v14 import (
    FeatureAdapter,
    ModerateDataTransform,
    MultimodalBIOT_V14,
    ShinMultimodalDataset,
    TokenLevelDelayAwareFNIRSToEEGCrossAttention,
    mixup_criterion,
    mixup_data,
    seed_worker,
)
from multimodal_v14.cli import build_parser
from multimodal_v14.training import (
    build_optimizer,
    load_partial_checkpoint,
    main,
    reset_module_parameters,
    set_module_trainable,
)

__all__ = [
    "FeatureAdapter",
    "ModerateDataTransform",
    "MultimodalBIOT_V14",
    "ShinMultimodalDataset",
    "TokenLevelDelayAwareFNIRSToEEGCrossAttention",
    "build_optimizer",
    "build_parser",
    "load_partial_checkpoint",
    "main",
    "mixup_criterion",
    "mixup_data",
    "reset_module_parameters",
    "seed_worker",
    "set_module_trainable",
]


if __name__ == "__main__":
    main(build_parser().parse_args())
