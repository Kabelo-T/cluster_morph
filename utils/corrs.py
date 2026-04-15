from multicam.multicam import MultiCAM  # type: ignore
from multicam.train import get_tt_indices  # type: ignore
from scipy.stats import spearmanr
import numpy as np


def build_mcam_corrs(spearm_corr: np.ndarray):
    """Build the correlation arrays with errors

    Parameters
    ----------
    spearm_corr : np.ndarray
        Spearman correlation array of shape (n_seeds, n_targets)
    """
    p25, p50, p75 = [], [], []

    for x in spearm_corr.T:
        p25.append(np.percentile(x, q=25))
        p50.append(np.percentile(x, q=50))
        p75.append(np.percentile(x, q=75))

    return p25, p50, p75


def multicorr(x: np.ndarray, y: np.ndarray, alpha: float = 0.0) -> tuple:
    """Calculate the Spearman correlation, RMSE, and R^2 between predicted and true values

    Parameters
    ----------
    x : np.ndarray
        features
    y : np.ndarray
        targets
    alpha : float, optional
        regularization strength for Ridge regression in MultiCAM, by default 0.0
    """
    n_targets = y.shape[1]
    n_features = x.shape[1]
    n_regions = x.shape[0]

    sp_corr = []
    weights = np.zeros((n_targets, n_features))

    r = 1000
    rmse = np.zeros(n_targets)
    r2 = np.zeros(n_targets)

    for seed in range(r):
        rng = np.random.default_rng(seed)
        train_idx, test_idx = get_tt_indices(
            n_points=n_regions, rng=rng, test_ratio=0.25
        )

        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        mv_model = MultiCAM(n_features, n_targets, alpha=alpha)
        mv_model.fit(x_train, y_train)

        coeff = mv_model.reg.coef_
        weights += coeff / r

        y_pred = mv_model.predict(x_test)

        sp_corr.append([
            spearmanr(yt, yp).correlation
            for yt, yp in zip(y_test.T, y_pred.T)
        ])

        error = y_test - y_pred
        rmse += np.sqrt(np.mean(error**2, axis=0)) / r

        ss_res = np.sum(error**2, axis=0)
        ss_tot = np.sum((y_test - np.mean(y_test, axis=0))**2, axis=0)

        # Guard against divide-by-zero if a target is constant
        r2_split = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)

        r2 += r2_split / r

    sp_corr = np.array(sp_corr)
    p25, p50, p75 = build_mcam_corrs(sp_corr)

    return sp_corr, p25, p50, p75, weights, rmse, r2
