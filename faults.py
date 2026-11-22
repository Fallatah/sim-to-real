"""Fault injection on sensor readings, and the cleaning/imputation strategies (CIS 656 Topic 3)."""
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler
import sim

TRAJ_LEN = 50  # transitions per adaptation trajectory


def collect(N, tp, unmodeled, rng):
    """Key-based (trajectory) sampling: K whole trajectories of L transitions, K*L = N.

    L = TRAJ_LEN (50) for N >= 50; for N < 50 a single trajectory of exactly N transitions.
    Each trajectory of L transitions logs L + NLAG + 1 readings: NLAG warm-up steps that
    supply the torque history of the first transition, and the reading after the last one.
    Nothing beyond the budget is generated, so every preprocessing step (spike
    treatment, imputation) only ever sees the readings that the N transitions require.
    """
    L = min(TRAJ_LEN, N)
    K = int(np.ceil(N / L))
    assert K * L == N, f'budget {N} is not a whole number of {L}-transition trajectories'
    p = sim.params(K, **tp, unmodeled=unmodeled)
    Q, DQ, TAU = sim.rollout_logs(K, L + sim.NLAG, p, rng)
    return np.concatenate([Q, DQ], 2), TAU          # R: (K, L+NLAG+1, 4) readings


# ------------------------------------------------------------------ injection
def missing_mask(R, TAU, mech, rate, rng):
    """Mask (K, T+1, 2): reading of joint j (q_j and dq_j together) lost at step t."""
    K, T1, _ = R.shape
    if rate == 0:
        return np.zeros((K, T1, 2), bool)
    if mech == 'MCAR':
        return rng.random((K, T1, 2)) < rate
    if mech == 'MAR':      # depends on the observed commanded torque
        tau = np.concatenate([TAU, TAU[:, -1:]], 1)
        w = np.abs(tau) ** 2 + 1e-3
        p = np.clip(rate * w / w.mean(), 0, 0.95)
        return rng.random((K, T1, 2)) < p
    if mech == 'MNAR':     # depends on the (unseen) velocity itself
        score = np.abs(R[..., 2:]) + rng.normal(0, 0.25 * R[..., 2:].std(), (K, T1, 2))
        return score > np.quantile(score, 1 - rate)
    raise ValueError(mech)


def apply_mask(R, M):
    R = R.copy()
    for j in range(2):
        R[..., j][M[..., j]] = np.nan
        R[..., 2 + j][M[..., j]] = np.nan
    return R


def add_spikes(R, rate, rng, sd):
    """Velocity spikes of +-5 SD on a fraction `rate` of velocity readings."""
    R = R.copy()
    hit = rng.random(R[..., 2:].shape) < rate
    sign = rng.choice([-1, 1], R[..., 2:].shape)
    R[..., 2:] += hit * sign * 5 * sd[2:]
    return R, hit


def add_noise(R, level, rng, sd):
    return R + rng.normal(0, 1, R.shape) * level * sd


# ------------------------------------------------------------------ treatment
def winsorize(R):
    R = R.copy()
    for c in (2, 3):
        x = R[..., c]
        q1, q3 = np.nanpercentile(x, [25, 75])
        iqr = q3 - q1
        R[..., c] = np.clip(x, q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return R


def mad_discard(R, thr=3.5):
    R = R.copy()
    for c in (2, 3):
        x = R[..., c]
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med)) * 1.4826
        R[..., c][np.abs(x - med) / mad > thr] = np.nan
    return R


def _table(R, TAU):
    """Rows = time steps; cols = 4 readings + commanded torques (t, t-1) as auxiliary predictors."""
    K, T1, _ = R.shape
    tau = np.concatenate([TAU, TAU[:, -1:]], 1)
    tau_prev = np.concatenate([TAU[:, :1], TAU], 1)
    return np.concatenate([R, tau, tau_prev], 2).reshape(K * T1, 8)


def _lag_table(R, TAU):
    """Lag-aware table: readings at t plus all four readings at t-1 and t+1 (and torques).

    Gives the iterative imputer the temporal neighbours of each missing reading
    (post-review addition; not preregistered).
    """
    K, T1, _ = R.shape
    nan = np.full((K, 1, 4), np.nan)
    prev = np.concatenate([nan, R[:, :-1]], 1)
    nxt = np.concatenate([R[:, 1:], nan], 1)
    tau = np.concatenate([TAU, TAU[:, -1:]], 1)
    tau_prev = np.concatenate([TAU[:, :1], TAU], 1)
    return np.concatenate([R, prev, nxt, tau, tau_prev], 2).reshape(K * T1, 16)


def _untable(A, shape):
    return A[:, :4].reshape(shape)


def impute(R, TAU, strategy, seed=0):
    """Return list of imputed reading arrays (one, or five for MICE) and indicator mask."""
    miss = np.isnan(R[..., :2])                    # (K,T+1,2) per joint
    if strategy == 'listwise' or not np.isnan(R).any():
        return [R], miss
    A = _table(R, TAU)
    if strategy in ('median', 'median+ind'):
        out = SimpleImputer(strategy='median').fit_transform(A)
        return [_untable(out, R.shape)], miss
    if strategy == 'knn':
        sc = RobustScaler().fit(A)
        out = sc.inverse_transform(KNNImputer(n_neighbors=5).fit_transform(sc.transform(A)))
        return [_untable(out, R.shape)], miss
    if strategy == 'regression':
        out = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=seed).fit_transform(A)
        return [_untable(out, R.shape)], miss
    if strategy == 'mice':
        outs = [IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=10,
                                 random_state=seed * 10 + i).fit_transform(A) for i in range(5)]
        return [_untable(o, R.shape) for o in outs], miss
    if strategy == 'lagreg':                       # post-review: lag-aware iterative regression
        A = _lag_table(R, TAU)
        out = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=seed).fit_transform(A)
        return [_untable(out, R.shape)], miss
    if strategy == 'interp':                       # exploratory, not preregistered
        out = R.copy()
        t = np.arange(R.shape[1])
        for k in range(R.shape[0]):
            for c in range(4):
                x = out[k, :, c]
                ok = ~np.isnan(x)
                if ok.sum() == 0:
                    x[:] = np.nanmedian(R[..., c])
                elif (~ok).any():
                    x[~ok] = np.interp(t[~ok], t[ok], x[ok])
        return [out], miss
    raise ValueError(strategy)


def to_sets(Rs, TAU, N, miss=None, indicator=False):
    """Readings -> transition datasets truncated to the first N collected transitions."""
    sets = []
    extra = None
    for R in Rs:
        X, Y, bi, ti = sim.transitions(R[..., :2], R[..., 2:], TAU)
        assert len(X) == N, "collect() must produce exactly N transitions"
        ok = ~(np.isnan(X).any(1) | np.isnan(Y).any(1))
        sets.append((X[ok], Y[ok]))
        if indicator:
            extra = miss[bi, ti].astype(float)[ok]
    return sets, extra
