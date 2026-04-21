# PyTolerance

A Python package for 1D/3D tolerance stack-up and kinematics analysis. It combines core statistical engines with intuitive quality management concepts, making tolerance analysis more rigorous and accessible.

## Installation

```bash
pip install -r requirements.txt
python -m pip install -e .
```

## Features

- **Advanced Dimension Tolerances:** Define components by CPK, limits, standard deviations, and more. Use actual vendor data to fit distributions. Support tool wear modelling.
- **Space and Rotation:** Track kinematic error propagation across multiple axes (`RotationTol`) and automatically parse Abbe Error. Handle clearance-based dynamic tilts (`AssemblyPlayTol`).
- **Chain and Pareto Analysis:** Perform deep vector analysis. Chain variations and build Pareto tables of contributors to focus engineering effort where it matters most.
- **GD&T Compatibility:** Utilize numerical integration across dimensions for geometries like concentricity/position.
- **Table-Driven Interface:** Write stacks cleanly as tables/lists, making it human-readable for mechanical and quality engineers.
- **Reporting & Graphing:** Export ready-to-plot data for Pandas, or seamlessly view `plotly` or `matplotlib` charts.

## Quickstart

### Table-Driven Chain Construction

```python
from pytolerance import DimensionTol, RotationTol, ToleranceChain

headers = ['type', 'name', 'direction', 'intent', 'tolerance', 'spec']
chain = ToleranceChain.from_table(
    "Assembly",
    headers,
    [DimensionTol, "Base_X", 'x', 50.0, 0.2, "cpk=1.33, vendor=Foxconn"],
    [DimensionTol, "Part_Y", 'y', 10.0, 0.05, "stdev=0.01"],
    [RotationTol,  "Tilt_Err", 'z', 0.0, 0.1, "stdev=0.01"]
)

# Output top contributors
print(chain.get_pareto())
```

### Actual Data and Tool Wear Simulation

```python
import numpy as np
from scipy import stats
from pytolerance import DimensionTol

# Simulating Tool wear and measured FAI Data
wear_profile = stats.uniform(loc=0.0, scale=0.05)
dim = DimensionTol(name="Shaft_OD", intent=10.0, stdev=0.01, wear_dist=wear_profile)

real_fai = np.random.normal(10.03, 0.02, 50)
dim.add_data(label="Batch1", data=real_fai, fitting="kde")

# Get PDF at theoretical wear vs actual KDE
print("Theoretical Prob at 10.03: ", dim.pdf(10.03, mode='theory'))
print("Actual Prob at 10.03: ", dim.pdf(10.03, mode='actual'))
```

### Optimize Tolerances

```python
from pytolerance import DimensionTol, ToleranceChain

# Provide target requirements and let the engine solve for USL/LSL.
dim = DimensionTol(intent=10.0, stdev=0.05)
suggestion = dim.evaluate_tolerance(target_cpk=1.67, allow_mean_shift=True)
print(suggestion)
```
