from typing import Union, Optional, List, Dict, Any
import pandas as pd

def music_basis(
        counts_df: pd.DataFrame,
        clusters: pd.Series,
        samples: pd.Series,
        non_zero: bool = True
    ) -> Dict[str, Any]:
    # 1) Drop zero‐only genes
    if non_zero:
        counts_df = counts_df[counts_df.sum(axis=1) > 0]
    # 2) Build Theta
    df = counts_df.T.copy()
    df['celltype'], df['sample'] = clusters, samples
    rel_ab = (
        df.groupby(['celltype','sample'])
          .apply(lambda sub: sub
             .drop(columns=['celltype','sample'])
             .sum(axis=0)
           / sub
             .drop(columns=['celltype','sample'])
             .values.sum()
          )
    )
    Theta = rel_ab.T; Theta.columns.names = ['celltype','sample']
    # 3) M.theta
    M_theta = Theta.groupby(level='celltype', axis=1).mean()
    # 4) S: average library size per cell
    libsize = (
        df.groupby(['celltype','sample'])
          .apply(lambda sub: sub
             .drop(columns=['celltype','sample'])
             .sum(axis=1)
             .mean()
          )
    )
    S_mat = libsize.unstack(level='celltype')
    # 5) M.S
    M_S = S_mat.mean(axis=0)
    # 6) Design D
    D = M_theta.multiply(M_S, axis=1)
    # 7) Sigma
    Sigma = pd.DataFrame({
        ct: Theta.xs(ct, level='celltype', axis=1).var(axis=1, ddof=1)
        for ct in Theta.columns.get_level_values('celltype').unique()
    })
    return {'Disgn.mtx': D, 'S': S_mat, 'M.S': M_S, 'M.theta': M_theta, 'Sigma': Sigma}