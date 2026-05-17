"""Integration tests for TSL-51 module consistency."""

import pytest
import torch


class TestModelRegistryConsistency:
    """Verify src.core.models is the single source of truth."""

    def test_core_registry_has_all_models(self):
        from src.core.models import MODEL_REGISTRY

        expected = {"gru", "gru_small", "mlp", "mlpmodel", "mopgru", "hybrid", "ctc"}
        assert expected <= set(MODEL_REGISTRY.keys())

    def test_train_models_reexports_match_core(self):
        from src.core.models import MODEL_REGISTRY as core_registry
        from src.train.models import MODEL_CLASSES

        # Every model in train.models should point to the same class as core
        for name, cls in MODEL_CLASSES.items():
            assert name in core_registry
            assert cls is core_registry[name]

    def test_get_model_fallback(self):
        from src.core.models import get_model, GRUModel

        assert get_model("gru") is GRUModel
        assert get_model("nonexistent") is GRUModel

    def test_mlp_alias(self):
        from src.core.models import MLP, MLPModel

        assert MLP is MLPModel

    def test_model_classes_forward(self):
        from src.core.models import MODEL_REGISTRY

        batch, seq, feat, classes = 2, 10, 162, 51
        x_seq = torch.randn(batch, seq, feat)
        x_flat = torch.randn(batch, feat)

        for name, ModelCls in MODEL_REGISTRY.items():
            model = ModelCls(input_dim=feat, num_classes=classes)
            model.eval()
            with torch.no_grad():
                # GRU-based models accept sequences; MLP expects flat features
                if name in ("mlp", "mlpmodel"):
                    out = model(x_flat)
                elif name == "ctc":
                    # CTC returns per-timestep logits: (batch, seq, num_classes + 1)
                    out = model(x_seq)
                    assert out.shape == (batch, seq, classes + 1), f"{name} output shape mismatch"
                    continue
                else:
                    out = model(x_seq)
            assert out.shape == (batch, classes), f"{name} output shape mismatch"


class TestFeatureLevelsConsistency:
    """Verify feature dimensions are consistent across modules."""

    def test_core_and_data_feature_dims_match(self):
        from src.core.features import FEATURE_LEVELS
        from src.data.feature_extraction import FEATURE_DIMS

        # Both dictionaries should agree on all common keys
        for key in FEATURE_LEVELS:
            assert FEATURE_LEVELS[key] == FEATURE_DIMS[key], f"Mismatch for {key}"

        for key in FEATURE_DIMS:
            assert FEATURE_DIMS[key] == FEATURE_LEVELS[key], f"Mismatch for {key}"

    def test_finger_dimension(self):
        from src.core.features import FEATURE_LEVELS
        from src.data.feature_extraction import FEATURE_DIMS

        assert FEATURE_LEVELS["finger"] == 252
        assert FEATURE_DIMS["finger"] == 252


class TestImportPaths:
    """Verify all public import paths work correctly."""

    def test_import_from_src(self):
        import src

        assert hasattr(src, "__version__")
        assert hasattr(src, "GRUModel")
        assert hasattr(src, "FEATURE_LEVELS")
        assert hasattr(src, "MODEL_REGISTRY")

    def test_import_from_src_core(self):
        from src.core import GRUModel, MLPModel, FEATURE_LEVELS, MODEL_REGISTRY

        assert GRUModel is not None
        assert MLPModel is not None
        assert "basic" in FEATURE_LEVELS
        assert "gru" in MODEL_REGISTRY

    def test_import_from_src_train(self):
        from src.train import TrainingConfig, MODEL_CLASSES

        assert TrainingConfig is not None
        assert "gru" in MODEL_CLASSES

    def test_import_from_src_data(self):
        from src.data import FEATURE_LEVELS, load_tsl51_user_sign

        assert "basic" in FEATURE_LEVELS
        assert load_tsl51_user_sign is not None

    def test_import_from_src_utils(self):
        from src.utils import safe_mean, safe_std, validate_feature_vector

        assert safe_mean is not None
        assert safe_std is not None
        assert validate_feature_vector is not None
