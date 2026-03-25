import pandas as pd
from musicpy.iter import music_iter
from musicpy.basis import music_basis

def music_prop(
            bulk_df: pd.DataFrame,
            sc_counts: pd.DataFrame,
            sc_clusters: pd.Series,
            sc_samples: pd.Series,
            verbose: bool = True,
            iter_max: int = 1000,
            nu: float = 1e-4,
            eps: float = 0.01,
            centered: bool = False,
            normalize: bool = False
    ) -> dict:
    """
    Port of MuSiC::music_prop (non‐covariance path).

    Inputs:
      - bulk_df:       genes×bulk_samples count matrix (DataFrame)
      - sc_counts:     genes×cells single‐cell counts DataFrame
      - sc_clusters:   cell‐indexed Series of cell‐type labels
      - sc_samples:    cell‐indexed Series of sample/donor labels
      - markers,select_ct: passed to music_basis
      - iter_max,nu,eps,centered,normalize: passed to music_iter

    Returns a dict with:
      * Est.prop.allgene   (bulk_samples×celltypes DataFrame of p_nnls)
      * Est.prop.weighted  (… of p_weight)
      * Weight.gene        (genes×bulk_samples DataFrame of per‐gene weights)
      * r.squared.full     (bulk_samples Series)
      * Var.prop           (bulk_samples×celltypes DataFrame of var_p)
    """
    # 1) restrict to genes present in bulk
    bulk_genes = bulk_df.index[bulk_df.mean(axis=1) != 0]
    bulk_df = bulk_df.loc[bulk_genes]

    # 1.5) convert to relative abundance (like R’s relative.ab)
    bulk_df = bulk_df.div(bulk_df.sum(axis=0), axis=1)

    # 2) build single‐cell basis
    basis = music_basis(sc_counts, sc_clusters, sc_samples, non_zero=True)
    D = basis['Disgn.mtx']  # genes × celltypes
    M_S = basis['M.S']  # celltype Series
    Sigma = basis['Sigma']  # genes × celltypes

    # 3) intersect genes
    common = D.index.intersection(bulk_df.index)
    D = D.loc[common]
    Sigma = Sigma.loc[common]

    # 3.5) drop any cell-types with NaNs in Sigma or D, or missing M_S
    valid_ct = [
        ct for ct in D.columns
        if not D[ct].isna().any()
           and not Sigma[ct].isna().any()
           and not pd.isna(M_S[ct])
    ]
    D = D[valid_ct]
    Sigma = Sigma[valid_ct]  # DataFrame: genes × valid_ct
    M_S = M_S[valid_ct]

    # 4) prepare output containers
    celltypes = D.columns
    samples = bulk_df.columns
    Est_all = pd.DataFrame(index=samples, columns=celltypes, dtype=float)
    Est_wt = pd.DataFrame(index=samples, columns=celltypes, dtype=float)
    Weight = pd.DataFrame(index=common, columns=samples, dtype=float)
    R2 = pd.Series(index=samples, dtype=float)
    Var_p = pd.DataFrame(index=samples, columns=celltypes, dtype=float)

    # 5) loop over each bulk sample
    for samp in samples:
        Y = bulk_df[samp]
        # drop genes with zero counts for this sample
        nonzero = Y != 0
        Y_sub = Y[nonzero]
        D_sub = D.loc[nonzero]
        Sigma_sub = Sigma.loc[nonzero]
        if verbose:
            print(f"[music_prop] {samp}: {nonzero.sum()} genes")

        # run deconvolution
        res = music_iter(
            Y_sub, D_sub, M_S, Sigma_sub,
            iter_max=iter_max, nu=nu, eps=eps,
            centered=centered, normalize=normalize
        )

        # collect outputs
        Est_all.loc[samp] = res['p_nnls']
        Est_wt.loc[samp] = res['p_weight']
        Weight.loc[nonzero, samp] = res['weight_gene']
        R2[samp] = res['R_squared']
        Var_p.loc[samp] = res['var_p']

    return {
        'Est.prop.allgene': Est_all,
        'Est.prop.weighted': Est_wt,
        'Weight.gene': Weight,
        'r.squared.full': R2,
        'Var.prop': Var_p
    }