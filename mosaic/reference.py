import pandas as pd
import os
import gzip
import shutil

def process_barcode(cell_name: str) -> str:
    if '_' in cell_name:
        barcode = cell_name.split('_', 1)[1]
    else:
        barcode = cell_name

    # Reverse complement
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    rev_comp = ''.join(complement.get(base, base) for base in reversed(barcode))
    return rev_comp

def create_barcode_mapping(cluster_labels_file: str) -> pd.Series:
    labels = pd.read_csv(cluster_labels_file, sep="\t")
    processed_barcodes = labels["cellName"].apply(process_barcode)

    return pd.Series(
        labels["cluster_name"].values,
        index=processed_barcodes
    )


def sort_fragments(experiment_fragments: dict[str, str],
                   barcode_mapping: pd.Series,
                   output_dir: str,
                   sample_every: int = 100) -> dict[str, str]:

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    cell_types = barcode_mapping.unique()
    barcode_mapping = barcode_mapping.to_dict()
    handles = {ct: open(f"{output_dir}/{ct}_fragments.tsv", "w") for ct in cell_types}
    buffers = {ct: [] for ct in cell_types}

    try:
        for experiment_name, fragments_file in experiment_fragments.items():
            print(f"\nProcessing {experiment_name} ...")
            n_matched = n_unmatched = 0
            _open = gzip.open if fragments_file.endswith(".gz") else open

            with _open(fragments_file, "rt") as fh:
                for i, line in enumerate(fh):
                    if line.startswith("#") or i % sample_every != 0:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 4:
                        continue

                    cell_type = barcode_mapping.get(parts[3].strip())
                    if cell_type:
                        buffers[cell_type].append(line)
                        n_matched += 1
                        if len(buffers[cell_type]) >= 10_000:
                            handles[cell_type].writelines(buffers[cell_type])
                            buffers[cell_type].clear()
                    else:
                        n_unmatched += 1

            total = n_matched + n_unmatched
            print(f"  Matched: {n_matched:,} | Unmatched: {n_unmatched:,} | "
                  f"Rate: {n_matched / total:.2%}" if total > 0 else "  No fragments matched.")

        for ct, buf in buffers.items():
            if buf:
                handles[ct].writelines(buf)
    finally:
        for h in handles.values():
            h.close()

    return {ct: f"{output_dir}/{ct}_fragments.tsv" for ct in cell_types}