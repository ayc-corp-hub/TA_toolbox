import numpy as np
from pytolerance import DimensionTol, RotationTol, AssemblyPlayTol, ToleranceChain

def test_tolerance_chain_linear():
    dim1 = DimensionTol(intent=10.0, stdev=0.1, direction='x')
    dim2 = DimensionTol(intent=5.0, stdev=0.05, direction='x')

    chain = ToleranceChain()
    chain += dim1
    chain += dim2

    samples = chain.rvs(size=1000, target_dir='x')
    assert np.isclose(np.mean(samples), 15.0, atol=0.05)

def test_tolerance_chain_rotation():
    chain = ToleranceChain()
    chain += RotationTol(intent=0.0, stdev=0.1, axis='z') # small random rotation
    chain += DimensionTol(intent=100.0, stdev=0.0, direction='x') # length 100 on x

    samples_y = chain.rvs(size=1000, target_dir='y')
    # Because of rotation, there should be some variation in Y
    assert np.std(samples_y) > 0

def test_tolerance_chain_assembly_play():
    outer_W = DimensionTol(intent=10.1, stdev=0.0)
    outer_L = DimensionTol(intent=20.0, stdev=0.0)
    inner_w = DimensionTol(intent=10.0, stdev=0.0)
    inner_l = DimensionTol(intent=15.0, stdev=0.0)

    play = AssemblyPlayTol(outer_W=outer_W, outer_L=outer_L, inner_w=inner_w, inner_l=inner_l, behavior='worst_case')
    max_angle = play.get_max_angle_deg()
    assert max_angle > 0

    samples = play.rvs_rad(size=10)
    assert np.max(np.abs(samples)) > 0

def test_tolerance_chain_from_table():
    headers = ['type', 'name', 'direction', 'intent', 'tolerance', 'spec']
    chain = ToleranceChain.from_table(
        "TestChain",
        headers,
        [DimensionTol, "PartA", 'x', 10.0, 0.2, "cpk=1.33"],
        [DimensionTol, "PartB", 'x', 5.0, 0.1, "stdev=0.02"]
    )

    assert len(chain.dimensions) == 2
    assert chain.dimensions[0].theory_mean == 10.0
    assert chain.dimensions[0].usl == 10.1
    assert chain.dimensions[0].lsl == 9.9
    assert chain.dimensions[1].theory_stdev == 0.02

def test_tolerance_chain_angle_variation():
    chain = ToleranceChain()
    chain += DimensionTol(intent=100.0, stdev=0.1, direction='x')
    chain += DimensionTol(intent=50.0, stdev=0.05, direction='y')

    angles = chain.evaluate_angle_variation(plane='xy', size=1000)
    assert len(angles) == 1000
    assert np.std(angles) > 0
