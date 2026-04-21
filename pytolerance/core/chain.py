import numpy as np
import pandas as pd
from ..utils.math_utils import parse_direction
from .base import AbstractTol
from ..engine.simulation import run_monte_carlo
from ..utils.reporter import generate_dataframe

def calculate_loop(components):
    """Calculates the 3D loop sum for nominal verification."""
    gap_vector = np.zeros(3)
    for dim in components:
        gap_vector += dim.nominal_vector
    return gap_vector

class ToleranceChain:
    def __init__(self, name="Master_Chain"):
        self.name = name
        self.components = []

    def __iadd__(self, other):
        if isinstance(other, AbstractTol):
            self.components.append(other)
        elif isinstance(other, ToleranceChain):
            self.components.extend(other.components)
        else:
            raise TypeError("Must add a valid Tolerance component or Chain")
        return self

    def add_dimension(self, dim):
        self += dim

    def __add__(self, other):
        new_chain = ToleranceChain(name=self.name)
        new_chain.components = self.components.copy()
        new_chain += other
        return new_chain

    def __getitem__(self, key):
        sub_chain = ToleranceChain(name=f"{self.name}_SubChain")
        if isinstance(key, slice):
            sub_chain.components = self.components[key]
        elif isinstance(key, list):
            sub_chain.components = [self.components[i] for i in key]
        elif isinstance(key, int):
            sub_chain.components = [self.components[key]]
        else:
            raise TypeError("Invalid index type")
        return sub_chain

    def filter_by(self, **kwargs):
        sub_chain = ToleranceChain(name=f"{self.name}_Filtered")
        for dim in self.components:
            match = True
            for k, v in kwargs.items():
                if getattr(dim, k, None) != v:
                    match = False
                    break
            if match:
                sub_chain += dim
        return sub_chain

    def get_nominal_gap(self):
        return calculate_loop(self.components)

    def check_closure(self, tolerance=1e-5):
        gap_vec = self.get_nominal_gap()
        gap_magnitude = np.linalg.norm(gap_vec)

        is_closed = gap_magnitude < tolerance
        print(f"--- Vector Loop Check ({self.name}) ---")
        print(f"Loop End Coordinate: [{gap_vec[0]:.4f}, {gap_vec[1]:.4f}, {gap_vec[2]:.4f}]")
        print(f"Error from Origin: {gap_magnitude:.4f}")
        print(f"Status: {'✅ Perfectly Closed' if is_closed else '⚠️ Not Closed (Open chain or design error)'}\n")
        return is_closed

    def get_endpoints(self, size=100000, mode='theory', condition='perfect', label=None):
        current_positions = np.zeros((size, 3))

        current_angles_rad_x = np.zeros(size)
        current_angles_rad_y = np.zeros(size)
        current_angles_rad_z = np.zeros(size)

        for item in self.components:
            if hasattr(item, 'rvs_rad'):
                rads = item.rvs_rad(size=size, mode=mode, condition=condition, label=label)
                if getattr(item, 'axis', 'z') == 'z':
                    current_angles_rad_z += rads
                elif getattr(item, 'axis', 'z') == 'x':
                    current_angles_rad_x += rads
                elif getattr(item, 'axis', 'z') == 'y':
                    current_angles_rad_y += rads
            else:
                length_samples = item.rvs(size=size, mode=mode, condition=condition, label=label)
                if np.isscalar(length_samples):
                    length_samples = np.full(size, length_samples)

                base_vec = getattr(item, 'unit_vector', np.array([1., 0., 0.]))

                cos_a = np.cos(current_angles_rad_z)
                sin_a = np.sin(current_angles_rad_z)

                global_dx = length_samples * (base_vec[0] * cos_a - base_vec[1] * sin_a)
                global_dy = length_samples * (base_vec[0] * sin_a + base_vec[1] * cos_a)
                global_dz = length_samples * base_vec[2]

                current_positions[:, 0] += global_dx
                current_positions[:, 1] += global_dy
                current_positions[:, 2] += global_dz

        return current_positions

    def run_analysis(self, samples=100000, target_dir='x'):
        target_vec = parse_direction(direction=target_dir)
        return run_monte_carlo(self.components, samples, target_vec)

    def rvs(self, size=100000, target_dir='x', mode='theory', condition='perfect', label=None):
        """Maintains previous API compatible execution"""
        return self.run_analysis(samples=size, target_dir=target_dir)

    def evaluate_angle_variation(self, plane='xy', size=100000, mode='theory', condition='perfect', label=None):
        nominal_vec = self.get_nominal_gap()
        endpoints = self.get_endpoints(size=size, mode=mode, condition=condition, label=label)

        plane = plane.lower()
        if plane == 'xy':
            idx_v, idx_u = 1, 0  # (Y, X)
        elif plane == 'xz':
            idx_v, idx_u = 2, 0  # (Z, X)
        elif plane == 'yz':
            idx_v, idx_u = 2, 1  # (Z, Y)
        else:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")

        if np.hypot(nominal_vec[idx_u], nominal_vec[idx_v]) < 1e-9:
            raise ValueError(f"Nominal vector length in {plane.upper()} plane is ~0, cannot define baseline angle!")

        nominal_angle = np.arctan2(nominal_vec[idx_v], nominal_vec[idx_u])
        sim_angles = np.arctan2(endpoints[:, idx_v], endpoints[:, idx_u])

        delta_rad = (sim_angles - nominal_angle + np.pi) % (2 * np.pi) - np.pi
        return np.rad2deg(delta_rad)

    def _get_active_dimensions(self, target_dir_input):
        target_vec = parse_direction(direction=target_dir_input)
        active_dims = []
        for dim in self.components:
            if hasattr(dim, 'rvs_rad'): # skip rotation
                continue
            unit_vec = getattr(dim, 'unit_vector', parse_direction(direction=getattr(dim, 'direction', 'x')))
            c = np.dot(unit_vec, target_vec)
            if abs(c) > 1e-9:
                active_dims.append((dim, c))
        return active_dims

    def get_pareto(self, target_dir='x', mode='theory', condition='perfect'):
        active_dims = self._get_active_dimensions(target_dir)
        records = []
        total_variance = 0.0

        for dim, c in active_dims:
            if hasattr(dim, '_get_effective_params'):
                _, eff_stdev = dim._get_effective_params(mode=mode, condition=condition)
            else:
                eff_stdev = 0.0

            variance = (eff_stdev * abs(c)) ** 2
            total_variance += variance

            records.append({
                "Dimension": dim.name,
                "Vendor": getattr(dim, 'vendor', 'Unknown'),
                "Projection_Coef": c,
                "Stdev": eff_stdev,
                "Variance": variance
            })

        df = pd.DataFrame(records)
        if total_variance > 0:
            df['Contribution_%'] = (df['Variance'] / total_variance) * 100
        elif not df.empty:
            df['Contribution_%'] = 0.0

        if not df.empty:
            df = df.sort_values(by='Contribution_%', ascending=False).reset_index(drop=True)
        return df

    def optimize_top_contributor(self, target_cpk=None, target_dpmo=None, allow_mean_shift=True,
                                 target_dir='x', mode='theory', condition='perfect'):
        pareto_df = self.get_pareto(target_dir, mode, condition)
        if pareto_df.empty:
            print("No valid dimensions for optimization.")
            return None

        top_dim_name = pareto_df.iloc[0]['Dimension']
        contribution = pareto_df.iloc[0]['Contribution_%']

        top_dim = next((dim for dim in self.components if dim.name == top_dim_name), None)

        print(f"🔍 Optimization Target: [{top_dim_name}] (Contribution: {contribution:.1f}%)")

        if hasattr(top_dim, 'evaluate_tolerance'):
            suggestion = top_dim.evaluate_tolerance(
                target_cpk=target_cpk,
                target_dpmo=target_dpmo,
                allow_mean_shift=allow_mean_shift,
                mode=mode, condition=condition
            )
            return suggestion
        return None

    def generate_report(self, spec=None):
        return generate_dataframe(self.components, spec=spec)

    @classmethod
    def from_table(cls, name, columns, *rows):
        chain = cls(name=name)

        for row_idx, row in enumerate(rows):
            row_data = dict(zip(columns, row))

            TolClass = row_data.get('type')
            if not callable(TolClass):
                raise ValueError(f"Row {row_idx+1} type must be a callable class.")

            kwargs = {}
            if 'name' in row_data: kwargs['name'] = row_data['name']
            if 'intent' in row_data: kwargs['intent'] = float(row_data['intent'])

            if 'tolerance' in row_data and row_data['tolerance'] is not None and row_data['tolerance'] != '':
                tol_val = float(row_data['tolerance'])
                intent = kwargs.get('intent', 0.0)
                kwargs['limits'] = [intent + tol_val/2, intent - tol_val/2]

            if 'direction' in row_data: kwargs['direction'] = row_data['direction']

            spec_str = row_data.get('spec', "")
            if isinstance(spec_str, str) and spec_str:
                items = spec_str.split(',')
                for item in items:
                    if '=' in item:
                        key, val = item.split('=', 1)
                        key = key.strip().lower()
                        val = val.strip()

                        if val.lower() == 'true': val = True
                        elif val.lower() == 'false': val = False
                        else:
                            try: val = float(val)
                            except ValueError: pass

                        kwargs[key] = val

            chain += TolClass(**kwargs)

        return chain

    @property
    def dimensions(self):
        """Backward compatibility for existing tests."""
        return self.components

    @dimensions.setter
    def dimensions(self, val):
        self.components = val
