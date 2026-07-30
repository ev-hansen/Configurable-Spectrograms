"""Library for building configurable, memory-efficient spectrogram plots.

Submodules are split by concern: single-panel rendering (:mod:`plotting`),
batch/parallel orchestration (:mod:`batch_runner`, :mod:`generic_batch`),
CDF data access (:mod:`cdf_utils`), and FAST-instrument-specific plotting
and batch processing under :mod:`configurable_spectrograms.fast`.

Every module in this package uses absolute imports only (never relative
``from . import`` imports). ``pre_commit_hooks/run_doctests.py`` execs each
doctested file's reduced source into a standalone, unparented module object
with no ``__package__`` set, so a relative import would raise
``ImportError: attempted relative import with no known parent package`` the
moment the hook tries to run that file's doctests.
"""

__version__ = "0.0.1"
