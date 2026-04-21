import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from ..utils.math_utils import parse_direction
from .base import AbstractTol
from ..engine.backend import xp
from ..engine.statistics import get_effective_params, calculate_dpmo

class DimensionTol(AbstractTol):
    def __init__(self, name="Unnamed_Dim", **kwargs):
        super().__init__(name)

        # 1. Vector and direction setting
        self.unit_vector = parse_direction(
            direction=kwargs.get('direction'),
            angle=kwargs.get('angle'),
            vector=kwargs.get('vector')
        )
        self.direction = kwargs.get('direction', 'x')

        # 2. Theoretical Spec and Variation
        self.theory_mean = kwargs.get('mean', kwargs.get('intent', 0.0))
        self.usl = kwargs.get('USL', None)
        self.lsl = kwargs.get('LSL', None)
        if 'limits' in kwargs:
            self.usl, self.lsl = max(kwargs['limits']), min(kwargs['limits'])

        self.cpk = kwargs.get('cpk', None)
        self.theory_stdev = kwargs.get('stdev', 0.0) # Default to 0 for rigid body constant vectors

        # Derive standard deviation from CPK if necessary
        if self.theory_stdev == 0.0 and self.cpk is not None and self.usl is not None and self.lsl is not None:
            dist_to_spec = min(self.usl - self.theory_mean, self.theory_mean - self.lsl)
            self.theory_stdev = dist_to_spec / (3 * self.cpk)

        # Tool Wear / Systematic Shift Distribution
        self.wear_dist = kwargs.get('wear_dist', None)

        # Dynamic switch: ignore variation (for what-if analysis)
        self.ignore_variation = kwargs.get('ignore_variation', False)

        # Uncertainty and process variation parameters (default: perfect)
        self.process_shift_sigma = kwargs.get('shift', 0.0)
        self.grr = kwargs.get('grr', 0.0)
        self.alpha = 1.0 - kwargs.get('confidence_level', 0.95)

        # Actual profiles database
        self.actual_profiles = {}

        # Additional metadata
        self.vendor = kwargs.get('vendor', 'Internal')
        self.type = kwargs.get('type', 'General')
        self.sn = kwargs.get('sn', 'N/A')

    @property
    def nominal_vector(self):
        """Returns the nominal vector of this dimension in 3D space."""
        return self.theory_mean * self.unit_vector

    def toggle_variation(self, state: bool):
        """Dynamically turns on/off the variation."""
        self.ignore_variation = not state

    def add_data(self, label, data, fitting='normal', **metadata):
        data_arr = np.array(data)
        mean = np.mean(data_arr)
        stdev = np.std(data_arr, ddof=1) if len(data_arr) > 1 else 0.0
        fitting = fitting.lower()

        if fitting == 'kde':
            dist = stats.gaussian_kde(data_arr)
            is_kde = True
        elif fitting in ['concentricity', 'position', 'rayleigh']:
            loc, scale = stats.rayleigh.fit(data_arr, floc=0)
            dist = stats.rayleigh(loc=loc, scale=scale)
            is_kde = False
        else:
            dist = stats.norm(loc=mean, scale=stdev if stdev > 0 else 1.0)
            is_kde = False

        self.actual_profiles[label] = {
            'data': data_arr,
            'mean': mean,
            'stdev': stdev,
            'n': len(data_arr),
            'dist': dist,
            'is_kde': is_kde,
            'fitting': fitting,
            'metadata': metadata
        }

    def _get_effective_params(self, mode='theory', label=None, condition='perfect', spec=None):
        return get_effective_params(self, mode, label, condition, spec)

    def _build_dist(self, mode, label, condition, spec):
        eff_mean, eff_stdev = get_effective_params(self, mode, label, condition, spec)
        if eff_stdev <= 1e-9:
            # Degenerate distribution (constant)
            return type('ConstantDist', (), {'pdf': lambda self, x: np.where(np.isclose(x, eff_mean), np.inf, 0),
                                             'cdf': lambda self, x: np.where(x >= eff_mean, 1, 0),
                                             'rvs': lambda self, size=1: np.full(size, eff_mean)})()
        return stats.norm(loc=eff_mean, scale=eff_stdev)

    def _get_target(self, mode='theory', label=None):
        if mode == 'theory':
            return self._build_dist(mode, label, 'perfect', None), False
        elif mode == 'actual':
            if not self.actual_profiles:
                raise ValueError(f"Dimension {self.name} has no actual data.")
            if label is None:
                label = list(self.actual_profiles.keys())[-1]
            p = self.actual_profiles[label]
            return p['dist'], p['is_kde']
        else:
            raise ValueError("mode must be 'theory' or 'actual'")

    def pdf(self, x, mode='theory', label=None, condition='perfect', spec=None):
        if mode == 'actual' and self.actual_profiles:
            dist, is_kde = self._get_target(mode, label)
            if is_kde: return dist.evaluate(x)
            return dist.pdf(x)

        base_dist = self._build_dist(mode, label, condition, spec)
        if self.wear_dist is None or mode != 'theory':
            return base_dist.pdf(x)

        combined_samples = base_dist.rvs(size=50000) + self.wear_dist.rvs(size=50000)
        kde = stats.gaussian_kde(combined_samples)
        return kde.evaluate(x)

    def cdf(self, x, mode='theory', label=None, condition='perfect', spec=None):
        if mode == 'actual' and self.actual_profiles:
            dist, is_kde = self._get_target(mode, label)
            if is_kde:
                if np.isscalar(x): return dist.integrate_box_1d(-np.inf, x)
                return np.array([dist.integrate_box_1d(-np.inf, val) for val in x])
            return dist.cdf(x)

        base_dist = self._build_dist(mode, label, condition, spec)
        if self.wear_dist is None or mode != 'theory':
            return base_dist.cdf(x)

        # Approximate CDF using KDE for wear distribution
        combined_samples = base_dist.rvs(size=50000) + self.wear_dist.rvs(size=50000)
        kde = stats.gaussian_kde(combined_samples)
        if np.isscalar(x): return kde.integrate_box_1d(-np.inf, x)
        return np.array([kde.integrate_box_1d(-np.inf, val) for val in x])

    def rvs(self, size=1, mode='theory', label=None, condition='perfect', spec=None):
        eff_mean, eff_stdev = self._get_effective_params(mode, label, condition, spec)

        if self.ignore_variation or eff_stdev <= 1e-9:
            samples = np.full(size, eff_mean)
        else:
            if mode == 'actual' and self.actual_profiles:
                dist, is_kde = self._get_target(mode, label)
                if is_kde:
                    samples = dist.resample(size).flatten()
                else:
                    samples = dist.rvs(size=size)
            else:
                base_dist = self._build_dist(mode, label, condition, spec)
                samples = base_dist.rvs(size=size)

        if mode == 'theory' and self.wear_dist is not None:
            samples += self.wear_dist.rvs(size=size)

        return samples

    def DPMO(self, spec=None, mode='theory', label=None, condition='perfect'):
        return calculate_dpmo(self, spec, mode, label, condition)

    def get_cpk(self, mean, stdev, spec=None):
        usl = max(spec) if spec else self.usl
        lsl = min(spec) if spec else self.lsl
        if usl is not None and lsl is not None and stdev > 0:
            return min(usl - mean, mean - lsl) / (3 * stdev)
        return None

    def compare_report(self, spec=None):
        records = []

        # Theory
        records.append({
            "Dimension": self.name,
            "Source": "Design Theory",
            "Mean": round(self.theory_mean, 4),
            "Stdev": round(self.theory_stdev, 4),
            "Cpk": round(self.get_cpk(self.theory_mean, self.theory_stdev, spec), 2) if self.get_cpk(self.theory_mean, self.theory_stdev, spec) is not None else None,
            "DPMO": round(self.DPMO(spec=spec, mode='theory'), 1),
            "Data_Count": "N/A"
        })

        # Actuals
        for label, p in self.actual_profiles.items():
            record = {
                "Dimension": self.name,
                "Source": label,
                "Mean": round(p['mean'], 4),
                "Stdev": round(p['stdev'], 4),
                "Cpk": round(self.get_cpk(p['mean'], p['stdev'], spec), 2) if self.get_cpk(p['mean'], p['stdev'], spec) is not None else None,
                "DPMO": round(self.DPMO(spec=spec, mode='actual', label=label), 1),
                "Data_Count": p['n']
            }
            record.update(p.get('metadata', {}))
            records.append(record)

        return records

    def evaluate_tolerance(self, target_cpk=None, target_dpmo=None, allow_mean_shift=False,
                           mode='theory', label=None, condition='perfect'):
        if target_cpk is None and target_dpmo is None:
            raise ValueError("Must provide either target_cpk or target_dpmo")

        eff_mean, eff_stdev = self._get_effective_params(mode, label, condition)

        if target_dpmo is not None:
            defect_rate_one_side = (target_dpmo / 1_000_000) / 2.0
            z_score = stats.norm.ppf(1.0 - defect_rate_one_side)
            calc_target_cpk = z_score / 3.0
            target_cpk = calc_target_cpk if target_cpk is None else target_cpk

        required_delta = 3.0 * target_cpk * eff_stdev

        if allow_mean_shift and self.usl is not None and self.lsl is not None:
            optimal_mean = (self.usl + self.lsl) / 2.0
            shift_required = optimal_mean - eff_mean
        else:
            optimal_mean = eff_mean
            shift_required = 0.0

        new_usl = optimal_mean + required_delta
        new_lsl = optimal_mean - required_delta

        return {
            "Target_Cpk": round(target_cpk, 2),
            "Target_DPMO": target_dpmo if target_dpmo else round((1 - stats.norm.cdf(target_cpk*3))*2e6, 1),
            "Recommended_USL": round(new_usl, 4),
            "Recommended_LSL": round(new_lsl, 4),
            "Mean_Shift_Required": round(shift_required, 4),
            "Current_Stdev": round(eff_stdev, 4)
        }

    def plot_pdf(self, engine='plt', mode='theory', label=None, condition='perfect', spec=None):
        eff_mean, eff_stdev = self._get_effective_params(mode, label, condition, spec)
        if eff_stdev <= 1e-9:
            eff_stdev = 1e-3 # avoid zero division for plotting

        x = np.linspace(eff_mean - 4*eff_stdev, eff_mean + 4*eff_stdev, 500)
        y = self.pdf(x, mode, label, condition, spec)

        usl = max(spec) if spec else self.usl
        lsl = min(spec) if spec else self.lsl

        title = f"{self.name} PDF ({mode}, {condition})"

        if engine == 'plotly':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='PDF', fill='tozeroy'))
            if usl is not None:
                fig.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
            if lsl is not None:
                fig.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
            fig.update_layout(title=title, xaxis_title="Dimension Value", yaxis_title="Density")
            return fig

        elif engine == 'plt':
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, y, label='PDF', color='blue')
            ax.fill_between(x, y, alpha=0.2, color='blue')
            if usl is not None:
                ax.axvline(usl, color='red', linestyle='--', label='USL')
            if lsl is not None:
                ax.axvline(lsl, color='red', linestyle='--', label='LSL')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            return fig, ax
        else:
            raise ValueError("engine must be 'plt' or 'plotly'")
