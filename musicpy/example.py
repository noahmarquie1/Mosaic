from mosaic.deconvolve import nnls_deconvolve, elastic_net_deconvolve, nu_svr_deconvolve
from mosaic.evaluate import evaluate_deconvolution, get_true_proportions
from mosaic.reference import create_barcode_mapping
from mosaic.signature import *
from musicpy.prop import music_prop


def run_music_example():
    # 1. Create combined AnnData object
    sample01_data = snap.read(Path("binned_data/sample_1.h5ad"), backed="r+")
    sample02_data = snap.read(Path("binned_data/sample_2.h5ad"), backed="r+")
    print("Finished loading sample data.")

    adatas = [
        ("sample01", sample01_data),
        ("sample02", sample02_data)
    ]
    combined_data = snap.AnnDataSet(adatas=adatas, filename="combined_data/combined.h5ads")
    print("Finished combining sample data.")

    # 2. Filter low-quality peaks to reach 50,000 total peaks
    snap.pp.select_features(combined_data, n_features=50_000, inplace=True)
    combined_data = combined_data.to_adata()
    selected_data = combined_data[:, combined_data.var['selected']]
    print("Finished filtering peaks.")

    # 4. Build count matrix
    counts_dense = selected_data.X.toarray()
    count_matrix = pd.DataFrame(
        counts_dense.T,
        index=selected_data.var_names,
        columns=selected_data.obs_names,
    )
    print("Finished building count matrix.")

    # 5. Add cluster labels to count matrix
    barcode_mapping = create_barcode_mapping("cluster_labels.txt")
    barcode_mapping = barcode_mapping[~barcode_mapping.index.duplicated(keep='first')]
    selected_data.obs["cluster_label"] = barcode_mapping.reindex(selected_data.obs_names).fillna("Unknown")

    # 6. Make sc_samples and sc_clusters series
    sc_samples = selected_data.obs['sample']
    print("Finished adding sample labels.")
    sc_clusters = selected_data.obs['cluster_label']
    print("Finished adding cluster labels.")

    # 7. Build bulk mixture
    test_data = snap.read(Path("binned_data/sample_3.h5ad"), backed=None)
    test_data = test_data[:, selected_data.var_names]
    bulk_counts = test_data.X.sum(axis=0)
    mixture_matrix = pd.DataFrame(
        np.asarray(bulk_counts).T,
        index=selected_data.var_names,
        columns=['mixture_sample']
    )
    print("Finished building bulk mixture.")

    # 8. Estimate proportions with MuSiC
    prop = music_prop(mixture_matrix, count_matrix, sc_clusters, sc_samples, True, 100)
    print("Finished estimating proportions.")
    print(prop['Est.prop.allgene'].iloc[0])

    # 9. Evaluate deconvolution
    true_proportions = get_true_proportions(
        "sample_data/sample03_data/fragments/SRR13252436_fragments.tsv",
        barcode_mapping,
    )
    est_proportions = pd.Series(prop['Est.prop.allgene'].iloc[0])
    est_proportions = est_proportions.reindex(true_proportions.index, fill_value=0.0)
    evaluate_deconvolution(est_proportions, true_proportions)
