import pandas as pd
from scipy.optimize import nnls

def deconvolve(signature_matrix: pd.DataFrame,
               mixture_vector: pd.Series) -> pd.Series:

    m = mixture_vector.reindex(signature_matrix.index)
    f, residual = nnls(signature_matrix.values, m.values)

    if f.sum() > 0:
        f = f / f.sum()

    proportions = pd.Series(f, index=signature_matrix.columns)

    print("\nEstimated cell type proportions:")
    print("─" * 35)
    for cell_type, proportion in proportions.sort_values(ascending=False).items():
        bar = "█" * int(proportion * 40)
        print(f"  {cell_type:<25} {proportion:.4f}  {bar}")
    print("─" * 35)
    print(f"  {'Total':<25} {proportions.sum():.4f}")
    print(f"\n  Residual: {residual:.4f}")

    return proportions