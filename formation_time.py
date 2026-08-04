import argparse
import warnings

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer
from multicam.multicam import MultiCAM  # type: ignore

import utils.file as futils

warnings.filterwarnings('ignore')

SM_PARAMS = ['rhalf_circ', 'C', 'sersic_amplitude', 'A', 'm14', 'core_C']
SM_PARAM_LABELS = ['Half-light Radius',
                    'Concentration',
                    'Sersic Amplitude',
                    'Asymmetry',
                    r'$m_{14}$',
                    'Core Concentration']
DS_PARAMS = ['eta_200[3]', 'delta_200[4]', 'fm_200[5]', 'fm2_200[6]', '3d']
DS_LABELS = [r'$\eta_{200c}$', r'$\Delta_{200c}$',
             r'$f_{m, 200c}$', r'$f_{m2, 200c}$', r'$m_{14, 3D}$']

PARAM_LABELS = {'sm': SM_PARAM_LABELS, 'ds': DS_LABELS}

XGB_PARAM_SPACE = {'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [1, 2, 3],
                    'n_estimators': [100, 250, 500]}


def _train_test_split(features, am, state, stack):
    n_halos = len(am)
    halo_idx = np.arange(n_halos)
    train_idx, test_idx = train_test_split(
        halo_idx, test_size=0.25, random_state=state)

    params = SM_PARAMS if features == 'sm' else DS_PARAMS
    if stack:
        # features/targets are stacked 3x, one row per projection
        train_idx = np.concatenate(
            [train_idx, train_idx + n_halos, train_idx + 2 * n_halos])
        test_idx = np.concatenate(
            [test_idx, test_idx + n_halos, test_idx + 2 * n_halos])

    return params, train_idx, test_idx


def predict_formation_time(features, m, state, stack=False):
    mah_dict, ma, am, aexp_bins, mass_bins, df = futils.load(features, stack=stack)
    params, train_idx, test_idx = _train_test_split(features, am, state, stack)
    y = np.vstack([am] * 3) if stack else am

    x = df[params].to_numpy()
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_features = x_train.shape[1]
    n_targets = y_train.shape[1]
    mv_model = MultiCAM(n_features, n_targets, alpha=0.1)
    mv_model.fit(x_train, y_train)
    y_pred = mv_model.predict(x_test)

    m_idx = np.argmin(np.abs(mass_bins - m))
    a_true = y_test[:, m_idx]
    a_pred = y_pred[:, m_idx]
    r2 = r2_score(a_true, a_pred)

    residuals = a_pred - a_true
    feature_corrs = [spearmanr(residuals, x_test[:, i]).correlation
                     for i in range(len(params))]

    return a_true, a_pred, r2, params, feature_corrs


def predict_formation_time_xgb(features, m, state, stack=False):
    mah_dict, ma, am, aexp_bins, mass_bins, df = futils.load(features, stack=stack)
    params, train_idx, test_idx = _train_test_split(features, am, state, stack)
    y = np.vstack([am] * 3) if stack else am

    x = df[params].to_numpy()
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    m_idx = np.argmin(np.abs(mass_bins - m))
    a_train = y_train[:, m_idx]
    a_true = y_test[:, m_idx]

    qt = QuantileTransformer(output_distribution='normal', random_state=state)
    x_train_qt = qt.fit_transform(x_train)
    x_test_qt = qt.transform(x_test)

    model = xgb.XGBRegressor(random_state=state)
    reg = GridSearchCV(model, XGB_PARAM_SPACE, cv=5, scoring='r2', verbose=0)
    reg.fit(x_train_qt, a_train)
    a_pred = reg.predict(x_test_qt)
    r2 = r2_score(a_true, a_pred)

    print(f'-------- xgboost [{features.upper()}: {", ".join(params)}] --------')
    print(f'best params : {reg.best_params_}')
    print(f'r2 : {r2:.3f}')

    residuals = a_pred - a_true
    feature_corrs = [spearmanr(residuals, x_test[:, i]).correlation
                     for i in range(len(params))]

    return a_true, a_pred, r2, params, feature_corrs


def plot_formation_time_row(axs, mc_result, features, xgb_result=None):
    a_true, a_pred, r2, params, feature_corrs = mc_result
    residuals = a_pred - a_true
    param_labels = PARAM_LABELS[features]

    all_residuals = residuals
    if xgb_result is not None:
        _, a_pred_xgb, _, _, _ = xgb_result
        all_residuals = np.concatenate([residuals, a_pred_xgb - a_true])
    bin_edges = np.histogram_bin_edges(all_residuals, bins=20)

    x_pos = np.arange(len(params))
    axs[0].scatter(x_pos, feature_corrs, s=60, color='tab:blue', edgecolors='k',
                   linewidths=0.3, label=fr'MultiCAM ($R^2$={r2:.3f})')
    axs[1].hist(residuals, bins=bin_edges, color='tab:blue', edgecolor='k', alpha=0.7,
               label=fr'MultiCAM ($\sigma$={residuals.std():.3f})')

    if xgb_result is not None:
        a_true_xgb, a_pred_xgb, r2_xgb, _, corrs_xgb = xgb_result
        residuals_xgb = a_pred_xgb - a_true_xgb
        axs[0].scatter(x_pos, corrs_xgb, s=60, color='tab:orange', edgecolors='k',
                       linewidths=0.3, label=fr'XGBoost ($R^2$={r2_xgb:.3f})')
        axs[1].hist(residuals_xgb, bins=bin_edges, color='tab:orange',
                   histtype='step', linewidth=2,
                   label=fr'XGBoost ($\sigma$={residuals_xgb.std():.3f})')

    axs[0].axhline(0, color='k', linestyle='--', linewidth=1)
    axs[0].set_xticks(x_pos)
    axs[0].set_xticklabels(param_labels, rotation=45, ha='right')
    axs[0].set_ylabel(r'$\rho_{\mathrm{sp}} (\mathrm{Residual}, X)$')
    axs[0].set_title(f'X = {features.upper()}')
    axs[0].legend()
    axs[0].grid()

    axs[1].axvline(0, color='k', linestyle='--')
    axs[1].set_xlabel(r'Residual: Predicted $-$ True a(m)')
    axs[1].set_ylabel('Count')
    axs[1].legend()
    axs[1].grid()


def formation_time(m=0.5, state=5, save=False, xgb_compare=False, stack=False):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    title = 'MultiCAM vs XGBoost' if xgb_compare else 'MultiCAM'
    fig.suptitle(f'Predicted a(m={m}): {title}')

    for row, features in enumerate(['sm', 'ds']):
        use_stack = stack if features == 'sm' else False
        mc_result = predict_formation_time(features, m, state, use_stack)
        xgb_result = predict_formation_time_xgb(
            features, m, state, use_stack) if xgb_compare else None
        plot_formation_time_row(axs[row], mc_result, features, xgb_result)

    plt.tight_layout()
    if save:
        plt.savefig('plots/formation_time.pdf')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=float, default=0.5,
                        help="mass fraction m to evaluate formation time a(m) at")
    parser.add_argument('-s', '--save', action='store_true',
                        help="Save plot")
    parser.add_argument('--xgb', action='store_true',
                        help="Also overlay XGBoost predictions on top of MultiCAM")
    parser.add_argument('--stack', action='store_true',
                        help="Stack all 3 SM projections instead of using zx only")
    args = parser.parse_args()
    formation_time(args.m, save=args.save, xgb_compare=args.xgb, stack=args.stack)
