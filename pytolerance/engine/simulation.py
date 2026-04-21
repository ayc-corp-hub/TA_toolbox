import numpy as np
from .backend import xp, USE_GPU

def run_monte_carlo(components: list, samples: int = 100000, target_dir_vec: np.ndarray = None):
    """Execute core monte carlo summation along a target direction utilizing xp (GPU/CPU)."""
    if target_dir_vec is None:
        target_dir_vec = xp.array([1., 0., 0.])
    else:
        target_dir_vec = xp.array(target_dir_vec)

    current_positions = xp.zeros((samples, 3))

    current_angles_rad_x = xp.zeros(samples)
    current_angles_rad_y = xp.zeros(samples)
    current_angles_rad_z = xp.zeros(samples)

    for item in components:
        if hasattr(item, 'rvs_rad'):
            rads = xp.array(item.rvs_rad(size=samples))
            if getattr(item, 'axis', 'z') == 'z':
                current_angles_rad_z += rads
            elif getattr(item, 'axis', 'z') == 'x':
                current_angles_rad_x += rads
            elif getattr(item, 'axis', 'z') == 'y':
                current_angles_rad_y += rads
        else:
            length_samples = xp.array(item.rvs(size=samples))
            if xp.isscalar(length_samples) or length_samples.ndim == 0:
                length_samples = xp.full(samples, float(length_samples))

            base_vec = getattr(item, 'unit_vector', np.array([1., 0., 0.]))
            base_vec = xp.array(base_vec)

            # Simplified rotation (assuming mainly z-axis)
            cos_a = xp.cos(current_angles_rad_z)
            sin_a = xp.sin(current_angles_rad_z)

            global_dx = length_samples * (base_vec[0] * cos_a - base_vec[1] * sin_a)
            global_dy = length_samples * (base_vec[0] * sin_a + base_vec[1] * cos_a)
            global_dz = length_samples * base_vec[2]

            current_positions[:, 0] += global_dx
            current_positions[:, 1] += global_dy
            current_positions[:, 2] += global_dz

    # Move back to CPU if we calculated on GPU
    result = current_positions.dot(target_dir_vec)
    if USE_GPU:
        result = xp.asnumpy(result)

    return result
