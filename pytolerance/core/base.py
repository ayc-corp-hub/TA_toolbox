from abc import ABC, abstractmethod
import numpy as np
from ..engine.backend import xp

class AbstractTol(ABC):
    """Base contract for all tolerance objects."""

    def __init__(self, name: str):
        self.name = name
        self.ignore_variation = False
        self.actual_profiles = {}
        self.vendor = "Internal"
        self.type = "General"

    @property
    @abstractmethod
    def nominal_vector(self) -> np.ndarray:
        """Returns the nominal vector of this dimension in 3D space."""
        pass

    @abstractmethod
    def rvs(self, size: int = 1, mode: str = 'theory', label: str = None, condition: str = 'perfect'):
        """Forces subclasses to implement Monte Carlo sampling logic."""
        pass

    def toggle_variation(self, state: bool):
        """Toggle variation on or off."""
        self.ignore_variation = not state

    def add_data(self, label: str, data: list, fitting: str = 'normal', **metadata):
        """Binds real measurement data."""
        pass
