# fluctuant

AGN vs TDE classification from ZTF light curves. Two pipelines:

- **ASTROMER**: pretrained transformer encoder (MACHO weights) -> 7-statistic pooling -> Random Forest
- **Physics**: 16 hand-crafted light curve features -> Gradient Boosting

## Installation

Requires Python 3.9+ and TensorFlow 2.13.

```bash
pip install tensorflow==2.13.0 keras==2.13.1
pip install .
```

ASTROMER pretrained weights (~2.5 MB) are loaded from `weights/macho/` relative to the working directory. Copy or symlink the `weights/` directory before running the ASTROMER pipeline.

## Notebooks

| Notebook | Pipeline |
|---|---|
| [`notebooks/Astromer Pipeline.ipynb`](notebooks/Astromer%20Pipeline.ipynb) | ASTROMER embeddings + Random Forest |
| [`notebooks/Physics Pipeline.ipynb`](notebooks/Physics%20Pipeline.ipynb) | Physics features + Gradient Boosting |

## CLI

```bash
# ASTROMER pipeline
fluctuant astromer --n-agn 500 --n-tde 500 --output-dir results/

# Physics pipeline
fluctuant physics --n-agn 500 --n-tde 500 --output-dir results/
```

Run `fluctuant astromer --help` or `fluctuant physics --help` for all options.
