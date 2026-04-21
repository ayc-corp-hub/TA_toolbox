import numpy as np
from scipy import stats
from pytolerance import DimensionTol

def test_dimension_tol_basic():
    dim = DimensionTol(name="Test", intent=10.0, limits=[10.2, 9.8], cpk=1.33, direction='x')
    assert dim.theory_mean == 10.0
    assert dim.usl == 10.2
    assert dim.lsl == 9.8
    assert np.isclose(dim.theory_stdev, 0.2 / (3 * 1.33))

    samples = dim.rvs(size=1000)
    assert len(samples) == 1000
    assert np.isclose(np.mean(samples), 10.0, atol=0.1)

def test_dimension_tol_real_data():
    data = np.random.normal(10.05, 0.05, 100)
    dim = DimensionTol(name="Test", intent=10.0, limits=[10.2, 9.8], cpk=1.33)
    dim.add_data("VendorA", data, fitting="normal")

    samples_theory = dim.rvs(size=1000, mode='theory')
    samples_actual = dim.rvs(size=1000, mode='actual', label='VendorA')

    assert np.isclose(np.mean(samples_theory), 10.0, atol=0.05)
    assert np.isclose(np.mean(samples_actual), 10.05, atol=0.05)

def test_dimension_tol_wear_dist():
    wear_dist = stats.uniform(loc=0.0, scale=0.05)
    dim = DimensionTol(name="Shaft", intent=10.0, stdev=0.01, wear_dist=wear_dist)

    samples = dim.rvs(size=1000)
    assert np.isclose(np.mean(samples), 10.025, atol=0.05)

def test_dimension_tol_worst_case_grr():
    dim = DimensionTol(intent=10.0, limits=[10.2, 9.8], cpk=1.33, shift=1.5, grr=0.20)
    dpmo_perfect = dim.DPMO(condition='perfect')
    dpmo_worst = dim.DPMO(condition='worst_case')
    assert dpmo_worst > dpmo_perfect

def test_dimension_tol_evaluation():
    dim = DimensionTol(intent=10.0, stdev=0.05)
    suggestion = dim.evaluate_tolerance(target_cpk=1.67)
    assert suggestion['Target_Cpk'] == 1.67
    expected_delta = 3.0 * 1.67 * 0.05
    assert np.isclose(suggestion['Recommended_USL'], 10.0 + expected_delta)
    assert np.isclose(suggestion['Recommended_LSL'], 10.0 - expected_delta)
