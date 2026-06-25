import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

from fluctuant.pipelines import AstromerPipeline


# ASTROMER.models.SingleBandEncoder is stubbed in conftest.py when TF is absent.
# The stub returns random (50, 256) arrays so all shape-dependent code runs.


@pytest.fixture(scope='module')
def pipeline_generated():
    """Pipeline after generate_dataset only."""
    p = AstromerPipeline()
    p.generate_dataset(n_agn=30, n_tde=30)
    return p


@pytest.fixture(scope='module')
def pipeline_embedded():
    """Pipeline after generate → preprocess → generate_embeddings."""
    p = AstromerPipeline()
    p.generate_dataset(n_agn=30, n_tde=30)
    p.preprocess()
    p.generate_embeddings()
    return p


@pytest.fixture(scope='module')
def pipeline_trained():
    """Fully trained pipeline (runs once per module)."""
    p = AstromerPipeline()
    p.generate_dataset(n_agn=40, n_tde=40)
    p.preprocess()
    p.generate_embeddings()
    clf, splits = p.train_classifier()
    return p, clf, splits


# ---------------------------------------------------------------------------
# Encoder training
# ---------------------------------------------------------------------------

class TestEncoderTraining:
    def test_init_from_scratch(self):
        p = AstromerPipeline(pretrained_weights=None)
        assert p.model is not None

    def test_train_encoder_returns_model(self, tmp_path):
        p = AstromerPipeline(pretrained_weights=None)
        result = p.train_encoder(
            n_train=20, n_val=10, epochs=2, patience=2,
            save_path=str(tmp_path / 'weights'),
        )
        assert result is p.model

    def test_model_usable_after_training(self, tmp_path):
        p = AstromerPipeline(pretrained_weights=None)
        p.train_encoder(
            n_train=20, n_val=10, epochs=2, patience=2,
            save_path=str(tmp_path / 'weights'),
        )
        p.generate_dataset(n_agn=10, n_tde=10)
        p.preprocess()
        p.generate_embeddings()
        assert p.embeddings is not None


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

class TestDatasetGeneration:
    def test_light_curve_count(self, pipeline_generated):
        assert len(pipeline_generated.light_curves) == 60

    def test_label_count(self, pipeline_generated):
        assert len(pipeline_generated.labels) == 60

    def test_label_balance(self, pipeline_generated):
        labels = pipeline_generated.labels
        assert np.sum(labels == 0) == 30
        assert np.sum(labels == 1) == 30

    def test_light_curves_are_2d_three_column(self, pipeline_generated):
        for lc in pipeline_generated.light_curves:
            assert lc.ndim == 2 and lc.shape[1] == 3


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_times_zero_mean(self):
        p = AstromerPipeline()
        p.generate_dataset(n_agn=10, n_tde=10)
        p.preprocess()
        for lc in p.light_curves:
            assert abs(lc[:, 0].mean()) < 1e-8

    def test_magnitudes_zero_mean(self):
        p = AstromerPipeline()
        p.generate_dataset(n_agn=10, n_tde=10)
        p.preprocess()
        for lc in p.light_curves:
            assert abs(lc[:, 1].mean()) < 1e-8

    def test_errors_unchanged(self):
        p = AstromerPipeline()
        p.generate_dataset(n_agn=5, n_tde=5)
        original_errors = [lc[:, 2].copy() for lc in p.light_curves]
        p.preprocess()
        for orig, proc in zip(original_errors, p.light_curves):
            np.testing.assert_array_equal(orig, proc[:, 2])


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class TestEmbeddings:
    EXPECTED_DIM = 7 * 256  # 7 pooling statistics × 256 ASTROMER d_model (MACHO head_dim)

    def test_embedding_shape(self, pipeline_embedded):
        n = len(pipeline_embedded.light_curves)
        assert pipeline_embedded.embeddings.shape == (n, self.EXPECTED_DIM)

    def test_embeddings_are_finite(self, pipeline_embedded):
        assert np.all(np.isfinite(pipeline_embedded.embeddings))

    def test_embeddings_stored_on_instance(self, pipeline_embedded):
        assert pipeline_embedded.embeddings is not None


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------

class TestClassifierTraining:
    def test_returns_sklearn_classifier(self, pipeline_trained):
        _, clf, _ = pipeline_trained
        assert hasattr(clf, 'predict') and hasattr(clf, 'predict_proba')

    def test_split_sizes_sum_to_total(self, pipeline_trained):
        _, _, (X_tr, X_te, y_tr, y_te, _) = pipeline_trained
        assert len(X_tr) + len(X_te) == 80

    def test_proba_in_unit_interval(self, pipeline_trained):
        _, _, (_, _, _, _, y_prob) = pipeline_trained
        assert np.all(y_prob >= 0) and np.all(y_prob <= 1)

    def test_prediction_lengths_match(self, pipeline_trained):
        _, clf, (_, X_te, _, y_te, y_prob) = pipeline_trained
        y_pred = clf.predict(X_te)
        assert len(y_pred) == len(y_te) == len(y_prob)

    def test_classifier_stored_on_instance(self, pipeline_trained):
        p, _, _ = pipeline_trained
        assert hasattr(p, 'classifier')


# ---------------------------------------------------------------------------
# Multi-band embeddings
# ---------------------------------------------------------------------------

class TestMultiBandEmbeddings:
    # 2 bands × 7 statistics × 256 d_model + 4 cross-band color features
    EXPECTED_DIM = 2 * 7 * 256 + 4

    @pytest.fixture(scope='class')
    def pipeline_multiband(self):
        p = AstromerPipeline()
        p.generate_dataset(n_agn=20, n_tde=20, multiband=True)
        p.preprocess()
        p.generate_embeddings()
        return p

    def test_light_curves_are_dicts(self):
        p = AstromerPipeline()
        p.generate_dataset(n_agn=5, n_tde=5, multiband=True)
        assert isinstance(p.light_curves[0], dict)
        assert set(p.light_curves[0].keys()) == {'g', 'r'}

    def test_both_bands_are_3_column(self):
        p = AstromerPipeline()
        p.generate_dataset(n_agn=5, n_tde=5, multiband=True)
        for lc in p.light_curves:
            for band in ('g', 'r'):
                assert lc[band].ndim == 2 and lc[band].shape[1] == 3

    def test_embedding_shape(self, pipeline_multiband):
        n = len(pipeline_multiband.light_curves)
        assert pipeline_multiband.embeddings.shape == (n, self.EXPECTED_DIM)

    def test_embeddings_are_finite(self, pipeline_multiband):
        assert np.all(np.isfinite(pipeline_multiband.embeddings))

    def test_cross_band_features_vary(self, pipeline_multiband):
        # Last 4 columns are cross-band features; they should not all be identical.
        cross = pipeline_multiband.embeddings[:, -4:]
        assert cross.std(axis=0).sum() > 0

    def test_raw_light_curves_stored_for_color(self):
        p = AstromerPipeline()
        p.generate_dataset(n_agn=5, n_tde=5, multiband=True)
        p.preprocess()
        assert hasattr(p, '_raw_light_curves')

    def test_multiband_classifier_trains(self, pipeline_multiband):
        p = pipeline_multiband
        # Need enough samples for a stratified 70/30 split
        p2 = AstromerPipeline()
        p2.generate_dataset(n_agn=30, n_tde=30, multiband=True)
        p2.preprocess()
        p2.generate_embeddings()
        clf, _ = p2.train_classifier()
        assert hasattr(clf, 'predict')


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

class TestPlots:
    def test_visualize_embeddings_saves_file(self, pipeline_embedded, tmp_path):
        path = str(tmp_path / 'tsne.png')
        fig = pipeline_embedded.visualize_embeddings(save_path=path)
        assert os.path.exists(path)
        assert fig is not None
        plt.close('all')

    def test_plot_example_lightcurves_saves_file(self, pipeline_generated, tmp_path):
        path = str(tmp_path / 'examples.png')
        fig = pipeline_generated.plot_example_lightcurves(n_examples=2, save_path=path)
        assert os.path.exists(path)
        assert fig is not None
        plt.close('all')
