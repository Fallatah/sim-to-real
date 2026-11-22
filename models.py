"""World models: DR-pretrained ensemble, residual adaptation, system identification."""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
import sim

# ---------------------------------------------------------------- fast MLP forward
def mlp_forward(m, X):
    h = X
    for i, (W, b) in enumerate(zip(m.coefs_, m.intercepts_)):
        h = h @ W + b
        if i < len(m.coefs_) - 1:
            h = np.maximum(h, 0)
    return h


class Scalers:
    """Robust scaling fitted once on source data (no target information)."""
    def __init__(self, X, Y):
        self.x = RobustScaler().fit(X)
        self.ysd = Y.std(0)

    def sx(self, X):
        return (X - self.x.center_) / self.x.scale_

    def sy(self, Y):
        return Y / self.ysd

    def uy(self, Ys):
        return Ys * self.ysd


class BaseEnsemble:
    """5-member MLP ensemble pretrained on domain-randomized source data."""
    def __init__(self, X, Y, n_members=5, seed=0, max_iter=40):
        self.sc = Scalers(X, Y)
        Xs, Ys = self.sc.sx(X), self.sc.sy(Y)
        rng = np.random.default_rng(seed)
        self.members = []
        for k in range(n_members):
            idx = rng.integers(0, len(X), len(X))           # bagging
            m = MLPRegressor(hidden_layer_sizes=(128, 128), batch_size=512, max_iter=max_iter,
                             learning_rate_init=1e-3, random_state=seed + k, tol=1e-6)
            m.fit(Xs[idx], Ys[idx])
            self.members.append(m)

    def predict_members(self, X):
        Xs = self.sc.sx(X)
        return np.stack([self.sc.uy(mlp_forward(m, Xs)) for m in self.members])

    def predict(self, X):
        return self.predict_members(X).mean(0)


class DRModel:
    name = 'DR'
    def __init__(self, base):
        self.base = base

    def predict(self, X):
        return self.base.predict(X)


class ResidualModel:
    """Frozen base + small residual MLP fitted to target data.

    Hyperparameters fixed before any target data were seen.
    extra: optional extra input columns (missing indicators), zeros at test time.
    pooled: list of (X, Y) imputed datasets (MICE) -> one residual per set, predictions averaged.
    """
    name = 'Residual'
    def __init__(self, base, datasets, seed=0, extra=None):
        self.base = base
        self.nets = []
        self.converged = []                    # False if the optimizer hit max_iter
        self.n_extra = 0 if extra is None else extra.shape[1]
        for i, (X, Y) in enumerate(datasets):
            R = base.sc.sy(Y - base.predict(X))
            Xs = base.sc.sx(X)
            if extra is not None:
                Xs = np.concatenate([Xs, extra], 1)
            net = MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-2, max_iter=200, batch_size=min(200, len(X)),
                               learning_rate_init=1e-3, n_iter_no_change=10, tol=1e-5,
                               random_state=seed * 100 + i)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)   # recorded below instead
                net.fit(Xs, R)
            self.converged.append(bool(net.n_iter_ < net.max_iter))
            self.nets.append(net)

    def predict(self, X):
        Xs = self.base.sc.sx(X)
        if self.n_extra:
            Xs = np.concatenate([Xs, np.zeros((len(X), self.n_extra))], 1)
        r = np.mean([mlp_forward(n, Xs) for n in self.nets], 0)
        return self.base.predict(X) + self.base.sc.uy(r)

    def residual_only(self, Xs_full):
        """Residual correction in scaled units, input already scaled (for SHAP/ICE)."""
        return np.mean([mlp_forward(n, Xs_full) for n in self.nets], 0)


SYSID_BOUNDS = dict(m=(0.3, 3.0), d=(0.1, 5.0), f=(0.1, 5.0))


class SysIDModel:
    """Fit the single perturbed parameter by minimizing one-step prediction error."""
    name = 'SysID'
    def __init__(self, axis, X, Y, ysd):
        self.axis = axis
        if len(X) == 0:
            self.p = {}
            return
        loss = lambda v: np.mean(((sim.sim_predict_delta(X, {axis: v}) - Y) / ysd) ** 2)
        if axis == 'lat':
            vals = list(range(sim.NLAG + 1))
            self.p = {'lat': vals[int(np.argmin([loss(v) for v in vals]))]}
        else:
            lo, hi = SYSID_BOUNDS[axis]
            r = minimize_scalar(lambda lv: loss(np.exp(lv)), bounds=(np.log(lo), np.log(hi)),
                                method='bounded', options=dict(xatol=1e-3))
            self.p = {axis: float(np.exp(r.x))}

    def predict(self, X):
        return sim.sim_predict_delta(X, self.p)


class OracleModel:
    """True simulator as a one-step model on feature rows.

    Exact only when the true delay is within the NLAG-step torque history. With
    constrained=True and a longer delay it uses the oldest available torque: a
    history-constrained reference, labelled 'Oracle4' in the outputs, not an exact one.
    """
    name = 'Oracle'
    def __init__(self, p_scalar, unmodeled=False, constrained=False):
        self.p, self.unmodeled, self.constrained = p_scalar, unmodeled, constrained

    def predict(self, X):
        return sim.sim_predict_delta(X, self.p, self.unmodeled, clip_lat=self.constrained)


def exact_multistep_nrmse(p_scalar, unmodeled, Q, DQ, TAU, H=10, state_sd=None):
    """Exact reference with the true required torque history (read from the full logs).

    Re-simulates every H-step window with the true parameters and the true applied
    torques, so it is exact for any delay, including delays beyond NLAG.
    """
    B, T1, _ = Q.shape
    starts = np.arange(sim.NLAG, T1 - 1 - H)
    bi, ti = np.meshgrid(np.arange(B), starts, indexing='ij')
    bi, ti = bi.ravel(), ti.ravel()
    n = len(bi)
    lat = int(p_scalar.get('lat', 0))
    p = sim.params(n, m=p_scalar.get('m', 1.0), d=p_scalar.get('d', 1.0), f=p_scalar.get('f', 1.0),
                   lat=0, unmodeled=unmodeled)
    q, dq = Q[bi, ti].copy(), DQ[bi, ti].copy()
    for h in range(H):
        applied = TAU[bi, np.clip(ti + h - lat, 0, None)]
        q, dq = sim.step(q, dq, applied, p)
    true = np.concatenate([Q[bi, ti + H], DQ[bi, ti + H]], 1)
    pred = np.concatenate([q, dq], 1)
    if state_sd is None:
        state_sd = np.concatenate([Q, DQ], 2).reshape(-1, 4).std(0)
    err = (pred - true) / state_sd
    return float(np.sqrt((err ** 2).mean(0)).mean())


# ---------------------------------------------------------------- evaluation
def multistep_nrmse(model, Q, DQ, TAU, H=10, state_sd=None):
    """Open-loop H-step prediction from every valid start in clean test logs."""
    B, T1, _ = Q.shape
    starts = np.arange(sim.NLAG, T1 - 1 - H)
    bi, ti = np.meshgrid(np.arange(B), starts, indexing='ij')
    bi, ti = bi.ravel(), ti.ravel()
    q, dq = Q[bi, ti].copy(), DQ[bi, ti].copy()
    for h in range(H):
        t = ti + h
        hist = np.stack([TAU[bi, t - k] for k in range(sim.NLAG + 1)], 1)
        d = model.predict(sim.features(q, dq, hist))
        q, dq = q + d[:, :2], np.clip(dq + d[:, 2:], -sim.DQ_MAX, sim.DQ_MAX)
    true = np.concatenate([Q[bi, ti + H], DQ[bi, ti + H]], 1)
    pred = np.concatenate([q, dq], 1)
    if state_sd is None:
        state_sd = np.concatenate([Q, DQ], 2).reshape(-1, 4).std(0)
    err = (pred - true) / state_sd
    return float(np.sqrt((err ** 2).mean(0)).mean()), err


def onestep_metrics(model, X, Y, ysd):
    P = model.predict(X)
    E = (P - Y) / ysd
    rmse = float(np.sqrt((E ** 2).mean(0)).mean())
    mae = float(np.abs(E).mean(0).mean())
    r2 = float(np.mean(1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)))
    return rmse, mae, r2
