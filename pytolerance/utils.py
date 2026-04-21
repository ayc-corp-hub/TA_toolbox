import numpy as np

def parse_direction(direction=None, angle=None, vector=None):
    """
    Ensures all directions are parsed into 3D unit vectors.
    """
    if vector is not None:
        v = np.array(vector, dtype=float)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else np.array([1., 0., 0.])

    if angle is not None:
        # Assuming angle is in XY plane (Degree)
        rad = np.deg2rad(angle)
        return np.array([np.cos(rad), np.sin(rad), 0.0])

    if isinstance(direction, str):
        d = direction.lower()
        if d == 'x': return np.array([1., 0., 0.])
        if d == 'y': return np.array([0., 1., 0.])
        if d == 'z': return np.array([0., 0., 1.])
        if d == '-x': return np.array([-1., 0., 0.])
        if d == '-y': return np.array([0., -1., 0.])
        if d == '-z': return np.array([0., 0., -1.])

    # Default to X direction
    return np.array([1., 0., 0.])
