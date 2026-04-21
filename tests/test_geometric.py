import numpy as np
from pytolerance import DimensionTol, GeometricTol

def test_geometric_tol_concentricity():
    dim_x = DimensionTol(intent=0.0, stdev=0.05, direction='x')
    dim_y = DimensionTol(intent=0.0, stdev=0.05, direction='y')

    conc = GeometricTol('concentricity', dim_x=dim_x, dim_y=dim_y)

    samples = conc.rvs(size=1000)
    assert len(samples) == 1000
    assert np.all(samples >= 0)

    pdf_val = conc.pdf(0.1)
    assert pdf_val > 0

    pdf_val_zero = conc.pdf(-0.1)
    assert pdf_val_zero == 0.0

def test_geometric_tol_ignore_variation():
    dim_x = DimensionTol(intent=0.0, stdev=0.05, direction='x')
    dim_y = DimensionTol(intent=0.0, stdev=0.05, direction='y')
    conc = GeometricTol('concentricity', dim_x=dim_x, dim_y=dim_y)

    conc.toggle_variation(False)
    samples = conc.rvs(size=10)
    assert np.all(samples == 0.0)
