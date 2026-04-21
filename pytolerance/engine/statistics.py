import numpy as np
from scipy import stats

def get_effective_params(dim_obj, mode='theory', label=None, condition='perfect', spec=None):
    usl = max(spec) if spec else dim_obj.usl
    lsl = min(spec) if spec else dim_obj.lsl
    tolerance = (usl - lsl) if (usl is not None and lsl is not None) else (6 * max(dim_obj.theory_stdev, 1e-9))

    sigma_ms = (dim_obj.grr * tolerance) / 6.0 if dim_obj.grr > 0 else 0.0

    if mode == 'theory':
        mean = dim_obj.theory_mean
        stdev = dim_obj.theory_stdev

        if condition == 'worst_case':
            stdev = np.sqrt(stdev**2 + sigma_ms**2)
            shift_val = dim_obj.process_shift_sigma * stdev
            if usl is not None and lsl is not None:
                direction = 1 if (usl - mean) < (mean - lsl) else -1
                mean += direction * shift_val
            else:
                mean += shift_val

    elif mode == 'actual':
        if not dim_obj.actual_profiles:
            raise ValueError(f"Dimension {dim_obj.name} has no actual data. Add data using add_data().")
        if label is None:
            label = list(dim_obj.actual_profiles.keys())[-1]
        p = dim_obj.actual_profiles[label]
        mean, stdev, n = p['mean'], p['stdev'], p['n']

        if condition == 'perfect' and dim_obj.grr > 0:
            stdev = np.sqrt(max(1e-9, stdev**2 - sigma_ms**2))
        elif condition == 'worst_case':
            if n > 1:
                t_val = stats.t.ppf(1 - dim_obj.alpha/2, df=n-1)
                margin_of_error = t_val * (stdev / np.sqrt(n))

                if usl is not None and lsl is not None:
                    direction = 1 if (usl - mean) < (mean - lsl) else -1
                    mean += direction * margin_of_error
                else:
                    mean += margin_of_error

                chi2_val = stats.chi2.ppf(dim_obj.alpha/2, df=n-1)
                stdev = np.sqrt((n - 1) * (stdev**2) / chi2_val)

    return mean, stdev

def calculate_dpmo(dim_obj, spec=None, mode='theory', label=None, condition='perfect'):
    usl = max(spec) if spec else dim_obj.usl
    lsl = min(spec) if spec else dim_obj.lsl

    prob_defect = 0.0
    if usl is not None: prob_defect += (1.0 - dim_obj.cdf(usl, mode, label, condition, spec))
    if lsl is not None: prob_defect += dim_obj.cdf(lsl, mode, label, condition, spec)

    return prob_defect * 1_000_000
