import pandas as pd

def generate_dataframe(components, spec=None):
    all_records = []
    for i, dim in enumerate(components):
        if hasattr(dim, 'compare_report'):
            dim_records = dim.compare_report(spec=spec)
            for r in dim_records:
                r['Chain_Index'] = i
            all_records.extend(dim_records)
    return pd.DataFrame(all_records)
