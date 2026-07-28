"""Public package interface for pandas-eda-check."""

from .core import check
from .comparison import compare

__version__ = "0.3.0"

__all__ = ["check", "compare", "__version__"]
