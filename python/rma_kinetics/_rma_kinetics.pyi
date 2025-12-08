from numpy import float64
from numpy.typing import NDArray

# ============================================================================
# Solution class (returned from Model.solve())
# ============================================================================

class Solution:
    """Solution returned from model integration."""

    @property
    def ts(self) -> NDArray[float64]: ...
    """
    Get time points.
    """

    @property
    def plasma_rma(self) -> NDArray[float64]: ...
    """Get plasma RMA array (available for constitutive, tetoff, and chemogenetic models)."""

    @property
    def brain_rma(self) -> NDArray[float64]: ...
    """Get brain RMA array (available for constitutive, tetoff, and chemogenetic models)."""

    @property
    def tta(self) -> NDArray[float64]: ...
    """Get tTA array (available for tetoff and chemogenetic models)."""

    @property
    def plasma_dox(self) -> NDArray[float64]: ...
    """Get plasma dox array (available for dox, tetoff, and chemogenetic models)."""

    @property
    def brain_dox(self) -> NDArray[float64]: ...
    """Get brain dox array (available for dox, tetoff, and chemogenetic models)."""

    @property
    def dreadd(self) -> NDArray[float64]: ...
    """Get DREADD array (available for chemogenetic models)."""

    @property
    def peritoneal_cno(self) -> NDArray[float64]: ...
    """Get peritoneal CNO array (available for cno and chemogenetic models)."""

    @property
    def plasma_cno(self) -> NDArray[float64]: ...
    """Get plasma CNO array (available for cno and chemogenetic models)."""

    @property
    def brain_cno(self) -> NDArray[float64]: ...
    """Get brain CNO array (available for cno and chemogenetic models)."""

    @property
    def plasma_clz(self) -> NDArray[float64]: ...
    """Get plasma CLZ array (available for cno and chemogenetic models)."""

    @property
    def brain_clz(self) -> NDArray[float64]: ...
    """Get brain CLZ array (available for cno and chemogenetic models)."""

    def elapsed_time(self) -> float: ...
    """Returns the elapsed time in seconds."""
