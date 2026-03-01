from mosaic.deconvolve import deconvolve
from mosaic.evaluate import evaluate_deconvolution, get_true_proportions
from mosaic.reference import create_barcode_mapping
from mosaic.signature import *

TEST_PEAKS = 100_000
TEST_FRAGMENTS = 500_000
TEST = True

def run_example():
    universe = build_peak_universe([
        "peaks/sample01_filtered.narrowPeak",
        "peaks/sample02_filtered.narrowPeak"
    ])
    universe = universe.head(TEST_PEAKS)

    sample_fragments = {
        "sample01": "sample01_data/fragments/fragments.tsv",
        "sample02": "sample02_data/fragments/SRR13252435_fragments.tsv"
    }

    barcode_mapping = create_barcode_mapping("cluster_labels.txt")
    cell_types = barcode_mapping.unique()

    sorted_fragments = {ct: f"sorted_fragments/{ct}_fragments.tsv" for ct in cell_types}
    count_matrix = build_count_matrix(
        sorted_fragments,
        universe,
        max_fragments=TEST_FRAGMENTS if TEST else None
    )
    normalised = quantile_normalize(count_matrix)
    filtered = filter_below_median(normalised)

    cell_type_map = pd.Series({ct: ct for ct in filtered.columns})
    signature_matrix = build_signature_matrix(filtered, cell_type_map)
    print(signature_matrix.head())

    mixture_vector = build_mixture_vector(
        'sample03_data/fragments/SRR13252436_fragments.tsv',
        universe,
        signature_matrix,
        max_fragments=TEST_FRAGMENTS
    )
    print(mixture_vector)

    print("Shapes of matrices:")
    print(f"  Signature matrix: {signature_matrix.shape}")
    print(f"  Mixture vector: {mixture_vector.shape}")

    estimated_proportions = deconvolve(signature_matrix, mixture_vector)
    true_proportions = get_true_proportions(
        "sample03_data/fragments/SRR13252436_fragments.tsv",
        barcode_mapping,
        max_fragments=TEST_FRAGMENTS
    )
    true_proportions = true_proportions.reindex(estimated_proportions.index, fill_value=0.0)
    evaluate_deconvolution(estimated_proportions, true_proportions)