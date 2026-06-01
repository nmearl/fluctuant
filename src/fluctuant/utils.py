import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             average_precision_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import warnings


def plot_evaluation_curves(y_test, y_proba, save_path=None):
    """
    ROC and Precision-Recall curves side by side.

    Parameters
    ----------
    y_test : array-like
    y_proba : array-like
        Predicted probabilities for the positive class.
    save_path : str, optional
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    ax1.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    ax2.plot(recall, precision, linewidth=2,
             label=f'PR (AP = {avg_precision:.3f})')
    ax2.axhline(y=y_test.sum() / len(y_test), color='k',
                linestyle='--', linewidth=1, label='Random')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved curves to {save_path}")

    return fig


def compare_classifiers(X_train, X_test, y_train, y_test, n_jobs=-1):
    """
    Train RF, GBM, LR, and SVM; return per-classifier AUC metrics.

    Returns
    -------
    dict
        Keys are classifier names; values are dicts with 'classifier',
        'cv_auc_mean', 'cv_auc_std', 'test_auc', 'test_avg_precision',
        'y_proba'.
    """
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=n_jobs
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=n_jobs
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', probability=True, random_state=42
        )
    }

    results = {}

    print("\nComparing Classifiers:")
    print("=" * 70)

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")

        # Cross-validation
        cv_scores = cross_val_score(
            clf, X_train, y_train,
            cv=5, scoring='roc_auc', n_jobs=n_jobs
        )

        # Train and test
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)

        results[name] = {
            'classifier': clf,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'test_auc': auc,
            'test_avg_precision': avg_precision,
            'y_proba': y_proba
        }

        print(f"  CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Test ROC-AUC: {auc:.3f}")
        print(f"  Test Avg Precision: {avg_precision:.3f}")

    print("\n" + "=" * 70)

    # Find best
    best_name = max(results, key=lambda k: results[k]['test_auc'])
    print(f"\nBest classifier: {best_name} (AUC = {results[best_name]['test_auc']:.3f})")

    return results


def augment_lightcurves(light_curves, n_augmentations=2):
    """
    Augment light curves with noise, time shift, and magnitude offset.

    Parameters
    ----------
    light_curves : list of ndarray, shape (n_obs, 3)
    n_augmentations : int
        Number of augmented copies per input light curve.

    Returns
    -------
    list of ndarray
        Original plus augmented light curves.
    """
    augmented = []

    for lc in light_curves:
        # Original
        augmented.append(lc.copy())

        for _ in range(n_augmentations):
            aug_lc = lc.copy()

            # 1. Add random noise to magnitudes
            noise_scale = np.random.uniform(0.01, 0.03)
            aug_lc[:, 1] += np.random.normal(0, noise_scale, len(aug_lc))

            # 2. Time shift (preserving relative times)
            time_shift = np.random.uniform(-50, 50)
            aug_lc[:, 0] += time_shift

            # 3. Magnitude offset
            mag_offset = np.random.uniform(-0.1, 0.1)
            aug_lc[:, 1] += mag_offset

            augmented.append(aug_lc)

    return augmented


def analyze_feature_importance(classifier, feature_names=None, top_n=20):
    """
    Horizontal bar chart of feature importances from a tree-based classifier.

    Parameters
    ----------
    classifier : fitted estimator with ``feature_importances_``
    feature_names : list of str, optional
    top_n : int
    """
    if not hasattr(classifier, 'feature_importances_'):
        print("Classifier doesn't have feature_importances_ attribute")
        return None

    importances = classifier.feature_importances_

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importances[indices][::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14)
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def temporal_train_test_split(light_curves, labels, split_time_fraction=0.7):
    """
    Assign objects to train/test based on mean observation time.

    Parameters
    ----------
    light_curves : list of ndarray, shape (n_obs, 3)
        Times in column 0.
    labels : array-like
    split_time_fraction : float
        Fraction of the global time range used as the train/test boundary.

    Returns
    -------
    train_indices : ndarray
    test_indices : ndarray
    """
    # Find global time range
    all_times = np.concatenate([lc[:, 0] for lc in light_curves])
    time_min, time_max = all_times.min(), all_times.max()
    split_time = time_min + split_time_fraction * (time_max - time_min)

    train_indices = []
    test_indices = []

    for i, lc in enumerate(light_curves):
        # Check if light curve is predominantly before or after split
        mean_time = lc[:, 0].mean()

        if mean_time < split_time:
            train_indices.append(i)
        else:
            test_indices.append(i)

    print(f"\nTemporal split at t={split_time:.1f}")
    print(f"Train: {len(train_indices)} objects (early)")
    print(f"Test: {len(test_indices)} objects (late)")

    return np.array(train_indices), np.array(test_indices)


def plot_classifier_comparison(results, save_path=None):
    """
    Bar chart comparing CV and test AUC for each classifier.

    Parameters
    ----------
    results : dict
        Output of :func:`compare_classifiers`.
    save_path : str, optional
    """
    names = list(results.keys())
    cv_means = [results[n]['cv_auc_mean'] for n in names]
    cv_stds = [results[n]['cv_auc_std'] for n in names]
    test_aucs = [results[n]['test_auc'] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width / 2, cv_means, width, yerr=cv_stds,
           label='CV AUC (mean ± std)', alpha=0.8, capsize=5)
    ax.bar(x + width / 2, test_aucs, width,
           label='Test AUC', alpha=0.8)

    ax.set_ylabel('ROC AUC Score', fontsize=12)
    ax.set_title('Classifier Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_confusion_matrix_plot(y_test, y_pred, class_names=['AGN', 'TDE'],
                                 save_path=None):
    """Count and normalized confusion matrix side by side."""
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Raw counts
    im1 = ax1.imshow(cm, cmap='Blues', aspect='auto')
    ax1.set_xticks(np.arange(len(class_names)))
    ax1.set_yticks(np.arange(len(class_names)))
    ax1.set_xticklabels(class_names)
    ax1.set_yticklabels(class_names)
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=13)

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax1.text(j, i, str(cm[i, j]),
                     ha='center', va='center', fontsize=14)

    # Normalized
    im2 = ax2.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(np.arange(len(class_names)))
    ax2.set_yticks(np.arange(len(class_names)))
    ax2.set_xticklabels(class_names)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=13)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax2.text(j, i, f'{cm_normalized[i, j]:.2f}',
                     ha='center', va='center', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig