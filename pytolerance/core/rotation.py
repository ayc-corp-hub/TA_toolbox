import numpy as np
from scipy import stats
from .base import AbstractTol
from ..engine.backend import xp

class RotationTol(AbstractTol):
    def __init__(self, name="Unnamed_Rotation", axis='z', **kwargs):
        """
        Handles pure angular variation (e.g., CNC rotary axis, fixture tilt).
        axis: 'x', 'y', or 'z'
        """
        super().__init__(name)
        self.axis = axis.lower()
        self.type = kwargs.get('type', 'Rotary_Axis')
        self.vendor = kwargs.get('vendor', 'Internal')

        # Nominal angle and variation in degrees
        self.theory_mean_deg = kwargs.get('intent', 0.0)
        self.theory_stdev_deg = kwargs.get('stdev', 0.0)
        self.ignore_variation = kwargs.get('ignore_variation', False)

        if self.theory_stdev_deg == 0.0 and 'cpk' in kwargs and 'limits' in kwargs:
            usl, lsl = max(kwargs['limits']), min(kwargs['limits'])
            self.theory_stdev_deg = min(usl - self.theory_mean_deg, self.theory_mean_deg - lsl) / (3 * kwargs['cpk'])

    @property
    def nominal_vector(self):
        # Rotation doesn't translate
        return np.array([0., 0., 0.])

    def rvs(self, size=1, mode='theory', label=None, condition='perfect'):
        return self.rvs_rad(size, mode, condition, label)

    def rvs_rad(self, size=1, mode='theory', condition='perfect', label=None):
        """Generates random angle samples in radians."""
        if self.ignore_variation or self.theory_stdev_deg <= 1e-9:
            deg_samples = np.full(size, self.theory_mean_deg)
        else:
            deg_samples = stats.norm(loc=self.theory_mean_deg, scale=self.theory_stdev_deg).rvs(size)
        return np.deg2rad(deg_samples)
