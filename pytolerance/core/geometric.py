import numpy as np
from scipy.integrate import quad
from .base import AbstractTol
from ..engine.backend import xp

class GeometricTol(AbstractTol):
    """Handles GD&T geometric tolerances using numerical integration for arbitrary distributions."""

    def __init__(self, tol_type, spec=None, **kwargs):
        super().__init__(kwargs.get('name', f"GD&T_{tol_type.lower()}"))
        self.tol_type = tol_type.lower()
        self.spec = spec
        self.unit_vector = np.array([0., 0., 0.]) # GD&T result is scalar magnitude

        if self.tol_type in ['concentricity', 'position']:
            self._build_radial_tol(**kwargs)
        else:
            raise ValueError(f"Unsupported geometric tolerance type: {self.tol_type}")

    def _build_radial_tol(self, **kwargs):
        """Build radial tolerance. Must provide dim_x and dim_y which have .pdf() and .rvs()"""
        self.dim_x = kwargs.get('dim_x')
        self.dim_y = kwargs.get('dim_y')

        if not self.dim_x or not self.dim_y:
            raise ValueError("Concentricity or Position requires dim_x and dim_y objects")

    @property
    def nominal_vector(self):
        return np.array([0., 0., 0.])

    def rvs(self, size=100000, mode='theory', label=None, condition='perfect'):
        """Monte Carlo sampling: naturally supports any distribution"""
        if self.ignore_variation:
            return np.zeros(size)

        if self.tol_type in ['concentricity', 'position']:
            x_sim = self.dim_x.rvs(size, mode=mode, label=label, condition=condition)
            y_sim = self.dim_y.rvs(size, mode=mode, label=label, condition=condition)
            r_sim = np.sqrt(x_sim**2 + y_sim**2)

            # GD&T is usually a diametral zone
            return 2.0 * r_sim

    def pdf(self, val, mode='theory', label=None, condition='perfect'):
        """
        Numerical integration for GD&T error PDF (val is diametral error).
        """
        if self.tol_type in ['concentricity', 'position']:
            r = val / 2.0
            if r <= 0:
                return 0.0

            def integrand(theta):
                x = r * np.cos(theta)
                y = r * np.sin(theta)

                pdf_x = self.dim_x.pdf(x, mode=mode, label=label, condition=condition)
                pdf_y = self.dim_y.pdf(y, mode=mode, label=label, condition=condition)

                return r * pdf_x * pdf_y

            # Integration over theta from 0 to 2pi
            pdf_val, _ = quad(integrand, 0, 2 * np.pi, limit=100)

            # Change of variables: D = 2R, so f_D(d) = f_R(d/2) / 2
            return pdf_val / 2.0

    def toggle_variation(self, state: bool):
        self.ignore_variation = not state
