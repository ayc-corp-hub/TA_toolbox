import numpy as np
from .base import AbstractTol
from ..engine.backend import xp

class AssemblyPlayTol(AbstractTol):
    def __init__(self, name="Assembly_Play", axis='z', **kwargs):
        """
        Simulates random rotation error caused by assembly clearance (Kinematic Play).
        """
        super().__init__(name)
        self.axis = axis.lower()
        self.type = 'Assembly_Clearance'

        self.outer_W = kwargs.get('outer_W')
        self.outer_L = kwargs.get('outer_L')
        self.inner_w = kwargs.get('inner_w')
        self.inner_l = kwargs.get('inner_l')

        # 'uniform' or 'worst_case'
        self.behavior = kwargs.get('behavior', 'uniform')

    @property
    def nominal_vector(self):
        return np.array([0., 0., 0.])

    def rvs(self, size=1, mode='theory', label=None, condition='perfect'):
        return self.rvs_rad(size, mode, condition, label)

    def rvs_rad(self, size=1, mode='theory', condition='perfect', label=None):
        """
        Dynamically calculate max rotation based on samples of dimensions.
        """
        if self.ignore_variation:
            return np.zeros(size)

        Ow = self.outer_W.rvs(size, mode=mode, condition=condition, label=label)
        Ol = self.outer_L.rvs(size, mode=mode, condition=condition, label=label)
        iw = self.inner_w.rvs(size, mode=mode, condition=condition, label=label)
        il = self.inner_l.rvs(size, mode=mode, condition=condition, label=label)

        diag = np.sqrt(il**2 + iw**2)

        ratio_w = np.clip(Ow / diag, -1.0, 1.0)
        ratio_l = np.clip(Ol / diag, -1.0, 1.0)

        theta_w = np.arctan2(il, iw) - np.arccos(ratio_w)
        theta_l = np.arctan2(iw, il) - np.arccos(ratio_l)

        max_theta = np.minimum(theta_w, theta_l)
        max_theta = np.maximum(max_theta, 0.0) # lock at 0 if interference

        if self.behavior == 'worst_case':
            signs = np.random.choice([-1.0, 1.0], size=size)
            return max_theta * signs
        else:
            random_ratios = np.random.uniform(-1.0, 1.0, size)
            return max_theta * random_ratios

    def get_max_angle_deg(self, mode='theory'):
        Ow = self.outer_W.theory_mean
        Ol = self.outer_L.theory_mean
        iw = self.inner_w.theory_mean
        il = self.inner_l.theory_mean

        diag = np.sqrt(il**2 + iw**2)
        theta_w = np.arctan2(il, iw) - np.arccos(np.clip(Ow/diag, -1, 1))
        theta_l = np.arctan2(iw, il) - np.arccos(np.clip(Ol/diag, -1, 1))
        max_theta = max(min(theta_w, theta_l), 0.0)

        return np.rad2deg(max_theta)
