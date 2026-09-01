import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_python(source):
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class V14ModuleSplitTests(unittest.TestCase):
    def assert_python_succeeds(self, source):
        result = run_python(source)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_package_exposes_components_by_responsibility(self):
        self.assert_python_succeeds(
            "from multimodal_v14.modeling import ("
            "FeatureAdapter, MultimodalBIOT_V14, "
            "TokenLevelDelayAwareFNIRSToEEGCrossAttention); "
            "from multimodal_v14.data import ("
            "ModerateDataTransform, ShinMultimodalDataset, "
            "mixup_criterion, mixup_data, seed_worker); "
            "from multimodal_v14.training import ("
            "build_optimizer, load_partial_checkpoint, main, "
            "reset_module_parameters, set_module_trainable); "
            "from multimodal_v14.cli import build_parser"
        )

    def test_legacy_module_reexports_the_same_public_objects(self):
        self.assert_python_succeeds(
            "import run_multimodal_v14 as legacy; "
            "from multimodal_v14 import ("
            "ModerateDataTransform, MultimodalBIOT_V14, "
            "ShinMultimodalDataset, mixup_data); "
            "assert legacy.MultimodalBIOT_V14 is MultimodalBIOT_V14; "
            "assert legacy.ModerateDataTransform is ModerateDataTransform; "
            "assert legacy.ShinMultimodalDataset is ShinMultimodalDataset; "
            "assert legacy.mixup_data is mixup_data"
        )

    def test_cli_preserves_key_defaults(self):
        self.assert_python_succeeds(
            "from multimodal_v14.cli import build_parser; "
            "args = build_parser().parse_args([]); "
            "assert args.in_channels_eeg == 30; "
            "assert args.in_channels_nirs == 36; "
            "assert args.n_classes == 3; "
            "assert args.cross_alignment_mode == 'local'; "
            "assert args.cross_fusion_mode == 'logit'; "
            "assert args.stage1_epochs == 15; "
            "assert args.epochs == 50"
        )

    def test_subject_split_script_imports_the_v14_model(self):
        self.assert_python_succeeds(
            "import run_multimodal_v14_subject_split as subject_split; "
            "from multimodal_v14 import MultimodalBIOT_V14; "
            "assert subject_split.MultimodalBIOT_V14 is MultimodalBIOT_V14"
        )

    def test_subject_split_defaults_to_training_from_scratch(self):
        self.assert_python_succeeds(
            "from run_multimodal_v14_subject_split import build_parser; "
            "assert build_parser().parse_args([]).checkpoint_path == ''"
        )

    def test_dataset_preserves_legacy_files_attribute(self):
        from multimodal_v14.data import ShinMultimodalDataset

        with tempfile.TemporaryDirectory() as root_path:
            dataset = ShinMultimodalDataset(root_path)
        self.assertTrue(hasattr(dataset, "files"))
        self.assertEqual(dataset.files, [])

    def test_dataset_normalize_keeps_the_legacy_instance_signature(self):
        from multimodal_v14.data import ShinMultimodalDataset

        with tempfile.TemporaryDirectory() as root_path:
            dataset = ShinMultimodalDataset(root_path)
        signal = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        try:
            normalized = ShinMultimodalDataset.normalize(dataset, signal)
        except TypeError as error:
            self.fail(f"legacy normalize signature regressed: {error}")
        self.assertTrue(torch.allclose(normalized.mean(dim=1), torch.zeros(2), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
