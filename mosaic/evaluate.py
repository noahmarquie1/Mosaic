import numpy as np
import pandas as pd

def evaluate_deconvolution(estimated_proportions: pd.Series, true_proportions: pd.Series) -> dict:
    # Compute error metrics
    errors = estimated_proportions - true_proportions
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    rmse = np.sqrt(squared_errors.mean())
    mae = abs_errors.mean()
    max_error = abs_errors.max()

    # PCC
    if len(true_proportions) > 1 and true_proportions.std() > 0 and estimated_proportions.std() > 0:
        correlation = np.corrcoef(true_proportions.values, estimated_proportions.values)[0, 1]
    else:
        correlation = np.nan

    # Print comparison
    print("\n" + "=" * 70)
    print("DECONVOLUTION EVALUATION")
    print("=" * 70)
    print(f"{'Cell Type':<25} {'True':<10} {'Estimated':<10} {'Error':<10}")
    print("─" * 70)

    for ct in sorted(estimated_proportions.index, key=lambda x: true_proportions[x], reverse=True):
        t = true_proportions[ct]
        e = estimated_proportions[ct]
        err = e - t
        print(f"{ct:<25} {t:>9.4f}  {e:>9.4f}  {err:>+9.4f}")

    print("─" * 70)
    print(f"\nError Metrics:")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  MAE (Mean Absolute Error):      {mae:.4f}")
    print(f"  Max Absolute Error:             {max_error:.4f}")
    print(f"  Pearson Correlation:            {correlation:.4f}")
    print("=" * 70)

    return {
        'true_proportions': true_proportions,
        'estimated_proportions': estimated_proportions,
        'errors': errors,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'correlation': correlation
    }


def get_true_proportions(fragments_file: str,
                         barcode_mapping: pd.Series,
                         max_fragments: int = None) -> pd.Series:
    cell_type_counts = {}
    n = 0
    with open(fragments_file, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                barcode = parts[3]
                if barcode in barcode_mapping.index:
                    cell_type = barcode_mapping[barcode]
                    if isinstance(cell_type, pd.Series):
                        cell_type = cell_type.iloc[0]
                    cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1

            n += 1
            if n >= max_fragments:
                break

    total = sum(cell_type_counts.values())
    if total == 0:
        print("Warning: No matched barcodes found!")
        return pd.Series(dtype=float)

    return pd.Series(cell_type_counts) / total