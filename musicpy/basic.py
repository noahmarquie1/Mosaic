import numpy as np
from scipy.optimize import nnls

def music_basic(
        Y: np.ndarray,         # shape (G,)
        X: np.ndarray,         # shape (G, K)
        S: np.ndarray,         # shape (K,)
        Sigma: np.ndarray,     # shape (G, K)
        iter_max: int = 1000,
        nu: float   = 1e-4,
        eps: float  = 0.01
    ) -> dict:
    """
    Weighted NNLS solver from MuSiC.
    Returns a dict with:
      - p_nnls: normalized NNLS solution (length K)
      - q_nnls: raw NNLS solution
      - fit_nnls: X @ q_nnls
      - resid_nnls: Y - fit_nnls
      - p_weight: normalized weighted solution
      - q_weight: raw weighted solution
      - fit_weight: X @ q_weight
      - resid_weight: Y - X @ q_weight
      - weight_gene: per-gene weights
      - converge: message
      - rsd: residuals from final weighted fit
      - R_squared: explained variance of final weighted fit
      - var_p: variance of p_weight estimates
    """
    G, K = X.shape

    # 1) Initial NNLS
    q_nnls, _ = nnls(X, Y)
    fit_nnls = X @ q_nnls
    resid_nnls = Y - fit_nnls

    # 2) Initial gene weights
    w = 1.0 / (nu + resid_nnls**2 + Sigma.dot((q_nnls * S)**2))
    sqrt_w = np.sqrt(w)

    # 3) First weighted fit
    Yw = Y * sqrt_w
    Xw = X * sqrt_w[:, None]
    q_weight, _ = nnls(Xw, Yw)
    p_weight = q_weight / q_weight.sum()
    #resid = Y - X @ q_weight
    resid_w = Yw - Xw @ q_weight

    # 4) Iterative re-weighting
    for i in range(1, iter_max+1):
        w = 1.0 / (nu + resid_w**2 + Sigma.dot((q_weight * S)**2))
        sqrt_w = np.sqrt(w)
        Yw = Y * sqrt_w
        Xw = X * sqrt_w[:, None]

        q_new, _ = nnls(Xw, Yw)
        p_new = q_new / q_new.sum()
        #resid_new = Y - X @ q_new
        resid_w = Yw - Xw @ q_new

        if np.sum(np.abs(p_new - p_weight)) < eps:
            q_weight = q_new
            p_weight = p_new
            #resid = resid_new
            converged = f"Converged at {i}"
            break

        q_weight = q_new
        p_weight = p_new
        #resid = resid_new
    else:
        converged = "Reached maxiter"

    # 5) Fit stats
    fitted = X @ q_weight
    R_squared = 1 - np.var(Y - fitted) / np.var(Y)
    # var_p: diag((Xw^T Xw)^-1) * mean(resid^2) / (sum(q_weight)^2)
    XtX_inv = np.linalg.inv(Xw.T @ Xw)
    var_p = np.diag(XtX_inv) * np.mean(resid_w**2) / (q_weight.sum()**2)

    return {
        "p_nnls":    q_nnls / q_nnls.sum(),
        "q_nnls":    q_nnls,
        "fit_nnls":  fit_nnls,
        "resid_nnls":resid_nnls,
        "p_weight":  p_weight,
        "q_weight":  q_weight,
        "fit_weight":fitted,
        "resid_weight": Y - X @ q_weight, #resid,
        "weight_gene": w,
        "converge":  converged,
        "rsd":       resid_w,
        "R_squared": R_squared,
        "var_p":     var_p,
    }