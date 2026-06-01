# Non-interactive backend — must be set before any other matplotlib import.
import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
from unittest.mock import MagicMock


def _build_astromer_stub():
    """Return a sys.modules-ready stub for ASTROMER.models.

    Produces random (50, 256) arrays so shape-dependent code runs correctly
    without a real TensorFlow installation.
    """
    rng = np.random.default_rng(0)

    mock_instance = MagicMock()
    mock_instance.from_pretraining.return_value = mock_instance
    # ASTROMER MACHO weights: head_dim=256 → d_model=256 (dff=128)
    mock_instance.encode.side_effect = lambda lcs: [
        rng.standard_normal((50, 256)) for _ in lcs
    ]

    mock_models = MagicMock()
    mock_models.SingleBandEncoder = MagicMock(return_value=mock_instance)
    return mock_models


# Only stub when TF is not actually importable (incomplete installation).
try:
    import tensorflow  # noqa: F401
except ImportError:
    sys.modules.setdefault('tensorflow', MagicMock())
    sys.modules.setdefault('keras', MagicMock())
    sys.modules.setdefault('ASTROMER', MagicMock())
    sys.modules['ASTROMER.models'] = _build_astromer_stub()
    sys.modules.setdefault('ASTROMER.preprocessing', MagicMock())
