import os
from pathlib import Path

import click


@click.group()
@click.version_option()
def main():
    """fluctuant: classify AGN variability vs TDEs using ZTF light curves."""
    pass


@main.command()
@click.option('--n-agn', default=500, show_default=True,
              help='Number of AGN light curves to generate.')
@click.option('--n-tde', default=500, show_default=True,
              help='Number of TDE light curves to generate.')
@click.option('--seed', default=42, show_default=True,
              help='Random seed.')
@click.option('--weights', default='macho', show_default=True,
              help='Pretrained ASTROMER weights (e.g. macho).')
@click.option('--tune/--no-tune', default=False, show_default=True,
              help='Tune Random Forest hyperparameters via grid search (slower).')
@click.option('--augment/--no-augment', default=False, show_default=True,
              help='Apply data augmentation (noise jitter, time shift, mag offset).')
@click.option('--output-dir', default='.', show_default=True, type=click.Path(),
              help='Directory for output plots.')
def astromer(n_agn, n_tde, seed, weights, tune, augment, output_dir):
    """Run the ASTROMER transformer embedding + Random Forest pipeline.

    Generates synthetic AGN and TDE light curves, encodes them with a
    pretrained ASTROMER model, and trains a Random Forest classifier on
    the multi-statistic pooled embeddings.
    """
    import numpy as np
    from fluctuant.pipelines import AstromerPipeline
    from fluctuant.utils import (plot_evaluation_curves,
                                  create_confusion_matrix_plot,
                                  analyze_feature_importance)

    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir)

    click.echo("=" * 60)
    click.echo("ASTROMER PIPELINE")
    click.echo("=" * 60)

    pipeline = AstromerPipeline(seed=seed, pretrained_weights=weights)

    click.echo("\n[1/5] Generating synthetic data...")
    pipeline.generate_dataset(n_agn=n_agn, n_tde=n_tde)

    if augment:
        from fluctuant.utils import augment_lightcurves
        click.echo("Applying data augmentation...")
        pipeline.light_curves = augment_lightcurves(
            pipeline.light_curves, n_augmentations=1
        )
        pipeline.labels = np.repeat(pipeline.labels, 2)
        click.echo(f"Augmented dataset: {len(pipeline.light_curves)} samples")

    click.echo("\n[2/5] Preprocessing...")
    pipeline.preprocess()

    click.echo("\n[3/5] Generating ASTROMER embeddings...")
    pipeline.generate_embeddings()

    click.echo("\n[4/5] Training classifier...")
    clf, (X_train, X_test, y_train, y_test, y_proba) = pipeline.train_classifier(
        tune_hyperparams=tune
    )

    click.echo("\n[5/5] Generating plots...")
    y_pred = clf.predict(X_test)

    pipeline.plot_example_lightcurves(n_examples=3,
                                      save_path=str(out / 'synthetic_examples.png'))
    plot_evaluation_curves(y_test, y_proba,
                           save_path=str(out / 'roc_pr_curves.png'))
    create_confusion_matrix_plot(y_test, y_pred,
                                 save_path=str(out / 'confusion_matrix.png'))
    pipeline.visualize_embeddings(save_path=str(out / 'embeddings_tsne.png'))

    if hasattr(clf, 'feature_importances_'):
        fig = analyze_feature_importance(clf, top_n=20)
        if fig is not None:
            fig.savefig(str(out / 'feature_importance.png'), dpi=300, bbox_inches='tight')

    click.echo(f"\nResults saved to {output_dir!r}:")
    for name in ('synthetic_examples.png', 'roc_pr_curves.png',
                 'confusion_matrix.png', 'embeddings_tsne.png',
                 'feature_importance.png'):
        click.echo(f"  {name}")


@main.command()
@click.option('--n-agn', default=500, show_default=True,
              help='Number of AGN light curves to generate.')
@click.option('--n-tde', default=500, show_default=True,
              help='Number of TDE light curves to generate.')
@click.option('--seed', default=42, show_default=True,
              help='Random seed.')
@click.option('--output-dir', default='.', show_default=True, type=click.Path(),
              help='Directory for output plots.')
def physics(n_agn, n_tde, seed, output_dir):
    """Run the physics-based light curve feature extraction pipeline.

    Extracts physically motivated features (amplitude, skewness, rise/decay
    timing, von Neumann ratio, etc.) and trains a Gradient Boosting classifier.
    No neural network required.
    """
    from fluctuant.classifiers import PhysicsBasedClassifier

    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir)

    click.echo("=" * 60)
    click.echo("PHYSICS-BASED PIPELINE")
    click.echo("=" * 60)

    clf_pipeline = PhysicsBasedClassifier(seed=seed)

    click.echo("\n[1/4] Generating synthetic data...")
    clf_pipeline.generate_dataset(n_agn=n_agn, n_tde=n_tde)

    click.echo("\n[2/4] Extracting physics features...")
    clf_pipeline.extract_features()

    click.echo("\n[3/4] Training classifier...")
    clf, (X_train, X_test, y_train, y_test, y_proba) = clf_pipeline.train()

    click.echo("\n[4/4] Generating diagnostic plots...")
    y_pred = clf.predict(X_test)

    clf_pipeline.analyze_features()
    clf_pipeline.plot_roc(y_test, y_proba,
                          save_path=str(out / 'roc_curve.png'))
    clf_pipeline.plot_confusion_matrix(y_test, y_pred,
                                       save_path=str(out / 'confusion_matrix.png'))
    clf_pipeline.plot_feature_space(save_path=str(out / 'feature_space_2d.png'))
    clf_pipeline.plot_feature_distributions(
        save_path=str(out / 'feature_distributions.png'))
    clf_pipeline.plot_pairwise_features(
        save_path=str(out / 'pairwise_features.png'))
    clf_pipeline.plot_example_predictions(
        save_path=str(out / 'example_predictions.png'))
    clf_pipeline.plot_feature_separability(
        save_path=str(out / 'feature_separability.png'))

    click.echo(f"\nResults saved to {output_dir!r}:")
    for name in ('roc_curve.png', 'confusion_matrix.png', 'feature_space_2d.png',
                 'feature_distributions.png', 'pairwise_features.png',
                 'example_predictions.png', 'feature_separability.png'):
        click.echo(f"  {name}")
