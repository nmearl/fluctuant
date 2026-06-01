import os

import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from fluctuant.classifiers import PhysicsBasedClassifier, extract_discriminative_features


@pytest.fixture(scope='module')
def single_tde_lc():
    from fluctuant.generators import SyntheticTransientGenerator
    gen = SyntheticTransientGenerator(seed=0)
    return gen.generate_tde(n_objects=1)[0]


@pytest.fixture(scope='module')
def trained_pipeline():
    """Trained PhysicsBasedClassifier (runs once per test module)."""
    p = PhysicsBasedClassifier(seed=42)
    p.generate_dataset(n_agn=100, n_tde=100)
    p.extract_features()
    clf, splits = p.train()
    return p, clf, splits


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestExtractDiscriminativeFeatures:
    def test_correct_feature_count(self, single_tde_lc):
        feats = extract_discriminative_features(single_tde_lc)
        assert len(feats) == len(PhysicsBasedClassifier.FEATURE_NAMES)

    def test_all_finite(self, single_tde_lc):
        feats = extract_discriminative_features(single_tde_lc)
        assert all(np.isfinite(f) for f in feats)

    def test_feature_names_count(self):
        assert len(PhysicsBasedClassifier.FEATURE_NAMES) == 16


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

class TestDatasetGeneration:
    def test_total_count(self):
        p = PhysicsBasedClassifier(seed=0)
        lcs, labels = p.generate_dataset(n_agn=20, n_tde=15)
        assert len(lcs) == 35
        assert len(labels) == 35

    def test_label_balance(self):
        p = PhysicsBasedClassifier(seed=0)
        lcs, labels = p.generate_dataset(n_agn=30, n_tde=30)
        assert np.sum(labels == 0) == 30
        assert np.sum(labels == 1) == 30

    def test_labels_are_shuffled(self):
        # With shuffling, labels should not all be 0s followed by all 1s.
        p = PhysicsBasedClassifier(seed=0)
        _, labels = p.generate_dataset(n_agn=100, n_tde=100)
        transitions = np.sum(np.diff(labels) != 0)
        assert transitions > 5


# ---------------------------------------------------------------------------
# Feature matrix extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_matrix_shape(self):
        p = PhysicsBasedClassifier(seed=0)
        p.generate_dataset(n_agn=20, n_tde=20)
        features = p.extract_features()
        assert features.shape == (40, 16)

    def test_no_nan_or_inf(self):
        p = PhysicsBasedClassifier(seed=0)
        p.generate_dataset(n_agn=20, n_tde=20)
        features = p.extract_features()
        assert np.all(np.isfinite(features))


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------

class TestClassifierTraining:
    def test_returns_classifier(self, trained_pipeline):
        _, clf, _ = trained_pipeline
        assert hasattr(clf, 'predict') and hasattr(clf, 'predict_proba')

    def test_split_sizes_sum_to_total(self, trained_pipeline):
        _, clf, (X_tr, X_te, y_tr, y_te, _) = trained_pipeline
        assert len(X_tr) + len(X_te) == 200

    def test_proba_valid_range(self, trained_pipeline):
        _, clf, (_, _, _, _, y_prob) = trained_pipeline
        assert np.all(y_prob >= 0) and np.all(y_prob <= 1)

    def test_auc_above_chance(self, trained_pipeline):
        _, clf, (_, _, _, y_te, y_prob) = trained_pipeline
        auc = roc_auc_score(y_te, y_prob)
        assert auc > 0.7, f"Expected AUC > 0.7, got {auc:.3f}"


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

class TestPlots:
    def test_plot_roc_saves_file(self, trained_pipeline, tmp_path):
        p, clf, (_, _, _, y_te, y_prob) = trained_pipeline
        path = str(tmp_path / 'roc.png')
        p.plot_roc(y_te, y_prob, save_path=path)
        assert os.path.exists(path)
        plt.close('all')

    def test_plot_confusion_matrix_saves_file(self, trained_pipeline, tmp_path):
        p, clf, (_, X_te, _, y_te, _) = trained_pipeline
        y_pred = clf.predict(X_te)
        path = str(tmp_path / 'cm.png')
        p.plot_confusion_matrix(y_te, y_pred, save_path=path)
        assert os.path.exists(path)
        plt.close('all')

    def test_plot_feature_space_saves_file(self, trained_pipeline, tmp_path):
        p, _, _ = trained_pipeline
        path = str(tmp_path / 'fs.png')
        p.plot_feature_space(save_path=path)
        assert os.path.exists(path)
        plt.close('all')

    def test_plot_feature_distributions_saves_file(self, trained_pipeline, tmp_path):
        p, _, _ = trained_pipeline
        path = str(tmp_path / 'fd.png')
        p.plot_feature_distributions(save_path=path)
        assert os.path.exists(path)
        plt.close('all')

    def test_plot_example_predictions_saves_file(self, trained_pipeline, tmp_path):
        p, _, _ = trained_pipeline
        path = str(tmp_path / 'ep.png')
        p.plot_example_predictions(n_examples=2, save_path=path)
        assert os.path.exists(path)
        plt.close('all')
