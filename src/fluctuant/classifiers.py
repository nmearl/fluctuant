"""Physics-based AGN vs TDE classifier using hand-crafted light curve features."""
from .generators import SyntheticTransientGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from scipy import stats


def extract_discriminative_features(lc):
    """
    Extract the 16 scalar features used to separate AGN from TDEs.

    Parameters
    ----------
    lc : ndarray, shape (n_obs, 3)
        Columns: time, magnitude, error.

    Returns
    -------
    list of float
        Length 16; see ``PhysicsBasedClassifier.FEATURE_NAMES`` for ordering.
    """
    times = lc[:, 0]
    mags = lc[:, 1]
    errs = lc[:, 2]

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    mags = mags[sort_idx]

    # Basic statistics
    mag_mean = mags.mean()
    mag_std = mags.std()
    mag_median = np.median(mags)
    amplitude = mags.max() - mags.min()

    # Percentiles
    p10, p25, p75, p90 = np.percentile(mags, [10, 25, 75, 90])
    iqr = p75 - p25

    # Asymmetry (TDEs are asymmetric: fast rise, slow decay)
    skewness = stats.skew(mags)

    # Peak properties
    peak_idx = np.argmin(mags)  # Brightest (lowest mag)
    peak_position = peak_idx / len(mags)  # Where in light curve (0=start, 1=end)

    # Rise vs decay time
    if peak_idx > 5 and peak_idx < len(mags) - 5:
        rise_time = times[peak_idx] - times[0]
        decay_time = times[-1] - times[peak_idx]
        rise_decay_ratio = rise_time / (decay_time + 1e-6)

        # Slopes
        rise_slope = (mags[0] - mags[peak_idx]) / (rise_time + 1e-6)
        decay_slope = (mags[-1] - mags[peak_idx]) / (decay_time + 1e-6)
    else:
        rise_decay_ratio = 1.0
        rise_slope = 0
        decay_slope = 0

    # Variability measures
    mag_diff = np.diff(mags)
    mean_abs_change = np.abs(mag_diff).mean()

    # Count sign changes (crossings of mean) - AGN oscillates more
    crossings = np.sum(np.diff(np.sign(mags - mag_mean)) != 0) / len(mags)

    # Late vs early behavior
    n_quarter = len(mags) // 4
    early_mean = mags[:n_quarter].mean()
    late_mean = mags[-n_quarter:].mean()
    early_late_diff = late_mean - early_mean

    # Monotonicity (TDEs more monotonic after peak)
    if peak_idx < len(mags) - 1:
        post_peak = mags[peak_idx:]
        # Fraction of observations that increase (get dimmer)
        monotonic_fraction = np.sum(np.diff(post_peak) > 0) / len(np.diff(post_peak))
    else:
        monotonic_fraction = 0.5

    # von Neumann ratio (randomness test)
    delta_sq = np.sum(mag_diff**2)
    variance = np.var(mags)
    von_neumann = delta_sq / (len(mags) * variance + 1e-10)

    # Concentration around peak
    bright_threshold = mags.min() + 0.3 * amplitude
    time_at_peak = np.sum(mags < bright_threshold) / len(mags)

    features = [
        amplitude,              # TDEs larger amplitude
        mag_std,
        iqr,
        skewness,              # TDEs more skewed
        peak_position,         # TDEs peak earlier
        rise_decay_ratio,      # TDEs: rise_decay_ratio << 1
        rise_slope,            # TDEs: steep rise
        decay_slope,           # TDEs: gradual decay
        mean_abs_change,
        crossings,             # AGN cross mean more
        early_late_diff,
        monotonic_fraction,    # TDEs more monotonic after peak
        von_neumann,           # AGN more stochastic (lower value)
        time_at_peak,          # TDEs spend more time at peak
        mag_mean,              # Overall brightness
        p90 - p10,            # Range
    ]

    return features


class PhysicsBasedClassifier:
    """Classifier using physical light curve features."""

    FEATURE_NAMES = [
        'amplitude', 'mag_std', 'iqr', 'skewness', 'peak_position',
        'rise_decay_ratio', 'rise_slope', 'decay_slope', 'mean_abs_change',
        'crossings', 'early_late_diff', 'monotonic_fraction',
        'von_neumann', 'time_at_peak', 'mag_mean', 'range_90_10',
    ]

    def __init__(self, seed=42):
        self.generator = SyntheticTransientGenerator(seed=seed)
        self.scaler = StandardScaler()

    def generate_dataset(self, n_agn=500, n_tde=500):
        print(f"Generating {n_agn} AGN and {n_tde} TDE...")

        agn_lcs = self.generator.generate_agn(
            n_objects=n_agn,
            include_flares=True,
            flare_fraction=0.5
        )

        tde_lcs = self.generator.generate_tde(n_objects=n_tde)

        self.light_curves = agn_lcs + tde_lcs
        self.labels = np.array([0] * n_agn + [1] * n_tde)

        # Shuffle
        idx = np.random.permutation(len(self.light_curves))
        self.light_curves = [self.light_curves[i] for i in idx]
        self.labels = self.labels[idx]

        return self.light_curves, self.labels

    def extract_features(self):
        """Extract features from all light curves."""
        print("Extracting discriminative features...")

        features = []
        for lc in self.light_curves:
            feat = extract_discriminative_features(lc)
            features.append(feat)

        self.features = np.array(features)
        print(f"Feature matrix: {self.features.shape}")

        # Check for NaN/Inf
        if np.any(~np.isfinite(self.features)):
            print("Warning: NaN/Inf detected, replacing with 0")
            self.features = np.nan_to_num(self.features)

        return self.features

    def train(self, test_size=0.3):
        """Train classifier."""
        print("\nTraining...")

        # Scale
        X_scaled = self.scaler.fit_transform(self.features)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.labels,
            test_size=test_size, random_state=42, stratify=self.labels
        )

        # Train
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        clf.fit(X_train, y_train)

        # CV score
        cv_scores = cross_val_score(clf, X_train, y_train,
                                     cv=5, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Test evaluation
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        print("\n" + "=" * 60)
        print(classification_report(y_test, y_pred,
                                    target_names=['AGN', 'TDE'], digits=3))

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(f"           Predicted")
        print(f"           AGN    TDE")
        print(f"Actual AGN {cm[0, 0]:4d}   {cm[0, 1]:4d}")
        print(f"       TDE {cm[1, 0]:4d}   {cm[1, 1]:4d}")

        auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC: {auc:.3f}")

        self.classifier = clf
        return clf, (X_train, X_test, y_train, y_test, y_proba)

    def plot_roc(self, y_test, y_proba, save_path='roc_curve.png'):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    def analyze_features(self):
        """Show most important features."""
        if not hasattr(self.classifier, 'feature_importances_'):
            return

        importances = self.classifier.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nTop 10 Features:")
        for i in range(min(10, len(self.FEATURE_NAMES))):
            idx = indices[i]
            print(f"{i+1}. {self.FEATURE_NAMES[idx]:20s}: {importances[idx]:.3f}")

    def plot_feature_space(self, save_path='feature_space_2d.png'):
        """PCA and t-SNE projections of the physics feature space."""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        print("\nGenerating feature space visualizations...")

        X_scaled = self.scaler.transform(self.features)

        # PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        colors = ['#3498db', '#e74c3c']
        class_labels = ['AGN', 'TDE']

        for i, (color, label) in enumerate(zip(colors, class_labels)):
            mask = self.labels == i
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=color, label=label, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)

        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
        ax1.set_title('PCA Projection', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)

        for i, (color, label) in enumerate(zip(colors, class_labels)):
            mask = self.labels == i
            ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=color, label=label, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)

        ax2.set_xlabel('t-SNE 1', fontsize=12)
        ax2.set_ylabel('t-SNE 2', fontsize=12)
        ax2.set_title('t-SNE Projection', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

        return fig

    def plot_feature_distributions(self, save_path='feature_distributions.png'):
        """Violin plots for the top-8 most important features, AGN vs TDE."""
        print("\nPlotting feature distributions...")

        # Get top 8 features by importance
        importances = self.classifier.feature_importances_
        top_indices = np.argsort(importances)[::-1][:8]

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes = axes.ravel()

        for i, feat_idx in enumerate(top_indices):
            ax = axes[i]

            agn_values = self.features[self.labels == 0, feat_idx]
            tde_values = self.features[self.labels == 1, feat_idx]

            # Violin plot
            parts = ax.violinplot([agn_values, tde_values],
                                  positions=[0, 1],
                                  showmeans=True, showmedians=True)

            # Color the violins
            for pc, color in zip(parts['bodies'], ['#3498db', '#e74c3c']):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['AGN', 'TDE'])
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(f'{self.FEATURE_NAMES[feat_idx]}\n(importance: {importances[feat_idx]:.3f})',
                        fontsize=10)
            ax.grid(alpha=0.3, axis='y')

        plt.suptitle('Feature Distributions: AGN vs TDE', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

        return fig

    def plot_pairwise_features(self, save_path='pairwise_features.png'):
        """Scatter plots for all pairs of the top-4 most important features."""
        print("\nPlotting pairwise feature relationships...")

        # Top 4 features
        importances = self.classifier.feature_importances_
        top_4 = np.argsort(importances)[::-1][:4]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()

        pair_idx = 0
        for i in range(len(top_4)):
            for j in range(i+1, len(top_4)):
                if pair_idx >= 6:
                    break

                ax = axes[pair_idx]
                feat_i, feat_j = top_4[i], top_4[j]

                agn_mask = self.labels == 0
                tde_mask = self.labels == 1

                ax.scatter(self.features[agn_mask, feat_i],
                          self.features[agn_mask, feat_j],
                          c='#3498db', label='AGN', alpha=0.5, s=20, edgecolors='k', linewidth=0.3)
                ax.scatter(self.features[tde_mask, feat_i],
                          self.features[tde_mask, feat_j],
                          c='#e74c3c', label='TDE', alpha=0.5, s=20, edgecolors='k', linewidth=0.3)

                ax.set_xlabel(self.FEATURE_NAMES[feat_i], fontsize=10)
                ax.set_ylabel(self.FEATURE_NAMES[feat_j], fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3)

                pair_idx += 1

        plt.suptitle('Pairwise Feature Relationships', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

        return fig

    def plot_example_predictions(self, n_examples=4, save_path='example_predictions.png'):
        """Show example light curves with predictions."""
        print("\nPlotting example predictions...")

        # Get predictions
        X_scaled = self.scaler.transform(self.features)
        y_proba = self.classifier.predict_proba(X_scaled)[:, 1]
        y_pred = self.classifier.predict(X_scaled)

        # Select examples: 2 correct, 2 misclassified
        correct_agn = np.where((self.labels == 0) & (y_pred == 0))[0]
        correct_tde = np.where((self.labels == 1) & (y_pred == 1))[0]
        wrong_agn = np.where((self.labels == 0) & (y_pred == 1))[0]  # AGN classified as TDE
        wrong_tde = np.where((self.labels == 1) & (y_pred == 0))[0]  # TDE classified as AGN

        indices = [
            correct_agn[0] if len(correct_agn) > 0 else 0,
            correct_tde[0] if len(correct_tde) > 0 else len(correct_agn),
            wrong_agn[0] if len(wrong_agn) > 0 else correct_agn[1] if len(correct_agn) > 1 else 0,
            wrong_tde[0] if len(wrong_tde) > 0 else correct_tde[1] if len(correct_tde) > 1 else len(correct_agn)
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        for i, idx in enumerate(indices):
            ax = axes[i]
            lc = self.light_curves[idx]

            true_label = 'AGN' if self.labels[idx] == 0 else 'TDE'
            pred_label = 'AGN' if y_pred[idx] == 0 else 'TDE'
            confidence = y_proba[idx] if y_pred[idx] == 1 else 1 - y_proba[idx]

            color = '#2ecc71' if true_label == pred_label else '#e74c3c'

            ax.errorbar(lc[:, 0], lc[:, 1], yerr=lc[:, 2],
                       fmt='o', markersize=4, alpha=0.6, capsize=2)
            ax.invert_yaxis()
            ax.set_xlabel('Time (days)', fontsize=11)
            ax.set_ylabel('Magnitude', fontsize=11)

            title = f"True: {true_label} | Pred: {pred_label} ({confidence:.2f})"
            ax.set_title(title, fontsize=11, fontweight='bold', color=color)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

        return fig

    def create_full_diagnostic_report(self, X_test, y_test, y_proba, save_prefix='diagnostic'):
        """Create comprehensive diagnostic report with all plots."""
        print("\nCreating comprehensive diagnostic report...")

        # 1. Feature space
        self.plot_feature_space(f'{save_prefix}_feature_space.png')

        # 2. Feature distributions
        self.plot_feature_distributions(f'{save_prefix}_distributions.png')

        # 3. Pairwise features
        self.plot_pairwise_features(f'{save_prefix}_pairwise.png')

        # 4. ROC curve
        self.plot_roc(y_test, y_proba, f'{save_prefix}_roc.png')

        # 5. Confusion matrix
        y_pred = self.classifier.predict(X_test)
        self.plot_confusion_matrix(y_test, y_pred, f'{save_prefix}_confusion.png')

        # 6. Example predictions
        self.plot_example_predictions(n_examples=4, save_path=f'{save_prefix}_examples.png')

        # 7. Feature separability analysis
        self.plot_feature_separability(f'{save_prefix}_separability.png')

        print(f"\nDiagnostic report saved with prefix: {save_prefix}_*")

    def plot_feature_separability(self, save_path='feature_separability.png'):
        """Cohen's d and Mann-Whitney U test for each feature, sorted by effect size."""
        from scipy.stats import mannwhitneyu

        print("\nAnalyzing feature separability...")

        # Calculate effect sizes and p-values for each feature
        effect_sizes = []
        p_values = []

        for i in range(self.features.shape[1]):
            agn_vals = self.features[self.labels == 0, i]
            tde_vals = self.features[self.labels == 1, i]

            # Effect size (Cohen's d)
            mean_diff = np.abs(agn_vals.mean() - tde_vals.mean())
            pooled_std = np.sqrt((agn_vals.std()**2 + tde_vals.std()**2) / 2)
            cohens_d = mean_diff / (pooled_std + 1e-10)
            effect_sizes.append(cohens_d)

            # Statistical test
            _, p_val = mannwhitneyu(agn_vals, tde_vals, alternative='two-sided')
            p_values.append(p_val)

        effect_sizes = np.array(effect_sizes)
        p_values = np.array(p_values)

        # Sort by effect size
        sorted_idx = np.argsort(effect_sizes)[::-1]

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Effect sizes
        colors = ['#2ecc71' if p < 0.001 else '#e74c3c' for p in p_values[sorted_idx]]
        ax1.barh(range(len(effect_sizes)), effect_sizes[sorted_idx], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(effect_sizes)))
        ax1.set_yticklabels([self.FEATURE_NAMES[i] for i in sorted_idx])
        ax1.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
        ax1.set_title("Feature Separability (green = p < 0.001)", fontsize=13, fontweight='bold')
        ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
        ax1.axvline(0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3, axis='x')

        # Feature importance vs separability
        if hasattr(self.classifier, 'feature_importances_'):
            importances = self.classifier.feature_importances_
            ax2.scatter(effect_sizes, importances, s=100, alpha=0.6, edgecolors='k')

            for i, name in enumerate(self.FEATURE_NAMES):
                if effect_sizes[i] > 0.5 or importances[i] > 0.05:
                    ax2.annotate(name, (effect_sizes[i], importances[i]),
                                fontsize=8, alpha=0.7, ha='right')

            ax2.set_xlabel("Cohen's d (Separability)", fontsize=12)
            ax2.set_ylabel('Feature Importance', fontsize=12)
            ax2.set_title('Importance vs Separability', fontsize=13, fontweight='bold')
            ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

        return fig

    def plot_confusion_matrix(self, y_test, y_pred, save_path='confusion_matrix.png'):
        """Enhanced confusion matrix visualization."""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Raw counts
        im1 = ax1.imshow(cm, cmap='Blues', aspect='auto')
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['AGN', 'TDE'])
        ax1.set_yticklabels(['AGN', 'TDE'])
        ax1.set_xlabel('Predicted', fontsize=12)
        ax1.set_ylabel('True', fontsize=12)
        ax1.set_title('Counts', fontsize=13, fontweight='bold')

        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, str(cm[i, j]),
                               ha='center', va='center', fontsize=16, fontweight='bold')

        # Normalized
        im2 = ax2.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['AGN', 'TDE'])
        ax2.set_yticklabels(['AGN', 'TDE'])
        ax2.set_xlabel('Predicted', fontsize=12)
        ax2.set_ylabel('True', fontsize=12)
        ax2.set_title('Normalized', fontsize=13, fontweight='bold')

        for i in range(2):
            for j in range(2):
                text = ax2.text(j, i, f'{cm_norm[i, j]:.2f}',
                               ha='center', va='center', fontsize=16, fontweight='bold')

        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

        return fig
