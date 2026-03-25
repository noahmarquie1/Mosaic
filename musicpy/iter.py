from typing import Dict

import pandas as pd
from musicpy.basic import music_basic
from typing import Dict, Any

def music_iter(
        Y: pd.Series,              # gene-indexed bulk vector
        D: pd.DataFrame,           # genes × celltypes'
        S: pd.Series,              # celltype-indexed sizes
        Sigma: pd.DataFrame,       # genes × celltypes
        iter_max: int = 1000,
        nu: float = 1e-4,
        eps: float = 0.01,
        centered: bool = False,
        normalize: bool = False
    ) -> Dict[str, Any]:
    # Align cell types
    common_ct = D.columns.intersection(S.index)
    D = D[common_ct]
    S = S[common_ct]
    Sigma = Sigma[common_ct] if isinstance(Sigma, pd.Series) else Sigma.loc[:, common_ct]

    # Align genes
    common_genes = Y.index.intersection(D.index)
    Y = Y.loc[common_genes]
    D = D.loc[common_genes]
    Sigma = Sigma.loc[common_genes]

    # Convert to arrays
    Y_arr = Y.values
    X = D.values
    S_arr = S.values
    Sigma_arr = Sigma.values

    # Centering / normalization
    if centered:
        X = X - X.mean()
        Y_arr = Y_arr - Y_arr.mean()
    if normalize:
        std_X = X.std()
        X = X / std_X
        S_arr = S_arr * std_X
        Y_arr = (Y_arr - Y_arr.mean()) / Y_arr.std()
    else:
        Y_arr = Y_arr * 100

    # Call core solver
    result = music_basic(Y_arr, X, S_arr, Sigma_arr, iter_max, nu, eps)

    # Attach labels back
    result['p_nnls'] = pd.Series(result['p_nnls'], index=D.columns)
    result['p_weight'] = pd.Series(result['p_weight'], index=D.columns)
    result['q_nnls'] = pd.Series(result['q_nnls'], index=D.columns)
    result['q_weight'] = pd.Series(result['q_weight'], index=D.columns)
    result['weight_gene'] = pd.Series(result['weight_gene'], index=D.index)
    result['R_squared'] = float(result['R_squared'])
    result['converge'] = result['converge']

    return result