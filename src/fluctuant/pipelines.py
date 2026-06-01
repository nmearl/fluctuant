from .generators import SyntheticTransientGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve,
                             average_precision_score)


class AstromerPipeline:
    """ASTROMER transformer embedding pipeline for AGN vs TDE classification."""

    def __init__(self, seed=42, pretrained_weights='macho'):
        from ASTROMER.models import SingleBandEncoder
        self.generator = SyntheticTransientGenerator(seed=seed)
        self.model = SingleBandEncoder()
        self.model = self.model.from_pretraining(pretrained_weights)
        self.embeddings = None
        self.labels = None

    def generate_dataset(self, n_agn=500, n_tde=500, cadence_days=3, duration_days=1000):
        """Generate synthetic AGN and TDE light curves."""
        print(f"Generating {n_agn} AGN and {n_tde} TDE light curves...")

        agn_lcs = self.generator.generate_agn(
            n_objects=n_agn,
            cadence_days=cadence_days,
            duration_days=duration_days,
            include_flares=True,
            flare_fraction=0.5,
        )
        tde_lcs = self.generator.generate_tde(
            n_objects=n_tde,
            cadence_days=cadence_days,
            duration_days=duration_days,
        )

        self.light_curves = agn_lcs + tde_lcs
        self.labels = np.array([0] * n_agn + [1] * n_tde)

        indices = np.random.permutation(len(self.light_curves))
        self.light_curves = [self.light_curves[i] for i in indices]
        self.labels = self.labels[indices]

        print(f"Total: {len(self.light_curves)} light curves")
        return self.light_curves, self.labels

    def preprocess(self):
        """Subtract per-light-curve mean from times and magnitudes for ASTROMER."""
        processed = []
        for lc in self.light_curves:
            times = lc[:, 0] - lc[:, 0].mean()
            mags = lc[:, 1] - lc[:, 1].mean()
            errs = lc[:, 2]
            processed.append(np.column_stack([times, mags, errs]))
        self.light_curves = processed
        return self.light_curves

    def generate_embeddings(self):
        """Extract multi-statistic feature vectors from ASTROMER encoder output."""
        print("Generating embeddings...")
        raw_embeddings = self.model.encode(self.light_curves)

        pooled = []
        for emb in raw_embeddings:
            mean_feat = emb.mean(axis=0)
            std_feat = emb.std(axis=0)
            temporal_grad = np.gradient(emb, axis=0).mean(axis=0)
            p25 = np.percentile(emb, 25, axis=0)
            p75 = np.percentile(emb, 75, axis=0)
            skew_approx = (mean_feat - np.median(emb, axis=0)) / (std_feat + 1e-8)
            early_mean = emb[:len(emb) // 3].mean(axis=0)
            late_mean = emb[-len(emb) // 3:].mean(axis=0)

            pooled.append(np.concatenate([
                mean_feat, std_feat, p25, p75,
                temporal_grad, skew_approx,
                early_mean - late_mean,
            ]))  # 7 × d_model features

        self.embeddings = np.vstack(pooled)
        print(f"Embeddings shape: {self.embeddings.shape}")
        return self.embeddings

    def train_classifier(self, test_size=0.3, tune_hyperparams=False):
        """Train a Random Forest classifier on scaled ASTROMER embeddings.

        Returns clf and (X_train, X_test, y_train, y_test, y_proba).
        X_train/X_test are already StandardScaler-transformed.
        """
        print("\nTraining classifier...")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.embeddings)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.labels,
            test_size=test_size, random_state=42, stratify=self.labels,
        )

        if tune_hyperparams:
            print("Tuning hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
            }
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1,
            )
            grid_search.fit(X_train, y_train)
            clf = grid_search.best_estimator_
            print(f"Best params: {grid_search.best_params_}")
        else:
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=20, random_state=42, n_jobs=-1,
            )
            clf.fit(X_train, y_train)

        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        print(f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        print("\n" + "=" * 60)
        print(classification_report(y_test, y_pred, target_names=['AGN', 'TDE'], digits=3))

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(f"           Predicted")
        print(f"           AGN    TDE")
        print(f"Actual AGN {cm[0, 0]:4d}   {cm[0, 1]:4d}")
        print(f"       TDE {cm[1, 0]:4d}   {cm[1, 1]:4d}")

        auc = roc_auc_score(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)
        print(f"\nROC AUC: {auc:.3f}")
        print(f"Avg Precision: {avg_prec:.3f}")

        self.classifier = clf
        self.scaler = scaler

        return clf, (X_train, X_test, y_train, y_test, y_proba)

    def visualize_embeddings(self, save_path=None):
        """t-SNE projection of ASTROMER embeddings, colored by class."""
        from sklearn.manifold import TSNE

        print("Generating t-SNE...")
        emb_2d = TSNE(n_components=2, random_state=42).fit_transform(self.embeddings)

        fig, ax = plt.subplots(figsize=(10, 8))
        for i, (color, label) in enumerate(zip(['#3498db', '#e74c3c'], ['AGN', 'TDE'])):
            mask = self.labels == i
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=color, label=label, alpha=0.6, s=30, edgecolors='k', linewidth=0.3)

        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title('ASTROMER Embeddings: AGN vs TDE', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig

    def plot_example_lightcurves(self, n_examples=3, save_path=None):
        """Plot example AGN and TDE light curves with model component overlays."""
        fig = self.generator.plot_with_model_overlay(n_examples=n_examples)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig
