"""Two-link planar arm (vertical plane) with controllable physics mismatch.

Source domain = nominal parameters. Target domain = one parameter changed
(link mass, joint damping, Coulomb friction or actuator latency), optionally
plus unmodeled effects the simulator's model class cannot express.
"""
import numpy as np

DT = 0.02          # control period (s)
SUB = 4            # integration substeps per control step
G = 9.81
L1, L2 = 0.5, 0.5
M1, M2 = 1.0, 0.8
DAMP0, FRIC0 = 0.3, 0.2
TAU_MAX = 12.0
NLAG = 4           # torque history in the features
DQ_MAX = 15.0

NOMINAL = dict(m=1.0, d=1.0, f=1.0, lat=0)   # scale factors / latency steps


def params(B, m=1.0, d=1.0, f=1.0, lat=0, unmodeled=False):
    """Per-trajectory parameter arrays."""
    full = lambda v: np.broadcast_to(np.asarray(v, float), (B,)).copy()
    return dict(m=full(m), d=full(d), f=full(f),
                lat=np.broadcast_to(np.asarray(lat, int), (B,)).copy(),
                unmodeled=bool(unmodeled))


def _accel(q, dq, tau, p):
    """Forward dynamics, batched. q, dq, tau: (B,2)."""
    ms = p['m']
    m1, m2 = M1 * ms, M2 * ms
    lc1, lc2 = L1 / 2, L2 / 2
    I1, I2 = m1 * L1 ** 2 / 12, m2 * L2 ** 2 / 12
    a = I1 + I2 + m1 * lc1 ** 2 + m2 * (L1 ** 2 + lc2 ** 2)
    b = m2 * L1 * lc2
    c = I2 + m2 * lc2 ** 2
    c2, s2 = np.cos(q[:, 1]), np.sin(q[:, 1])
    M11 = a + 2 * b * c2
    M12 = c + b * c2
    M22 = c
    cor1 = -b * s2 * (2 * dq[:, 0] * dq[:, 1] + dq[:, 1] ** 2)
    cor2 = b * s2 * dq[:, 0] ** 2
    g1 = (m1 * lc1 + m2 * L1) * G * np.cos(q[:, 0]) + m2 * lc2 * G * np.cos(q[:, 0] + q[:, 1])
    g2 = m2 * lc2 * G * np.cos(q[:, 0] + q[:, 1])
    damp = (DAMP0 * p['d'])[:, None] * dq
    fric = (FRIC0 * p['f'])[:, None] * np.tanh(dq / 0.05)
    rhs = tau - np.stack([cor1, cor2], 1) - np.stack([g1, g2], 1) - damp - fric
    if p['unmodeled']:
        # Stribeck bump + elastic cable on joint 2: outside the simulator's model class
        rhs -= 0.5 * np.exp(-(dq / 0.3) ** 2) * np.tanh(dq / 0.02)
        rhs[:, 1] -= 1.0 * np.sin(q[:, 1])
    det = M11 * M22 - M12 ** 2
    dd1 = (M22 * rhs[:, 0] - M12 * rhs[:, 1]) / det
    dd2 = (-M12 * rhs[:, 0] + M11 * rhs[:, 1]) / det
    return np.stack([dd1, dd2], 1)


def step(q, dq, tau_applied, p):
    h = DT / SUB
    for _ in range(SUB):
        dq = np.clip(dq + h * _accel(q, dq, tau_applied, p), -DQ_MAX, DQ_MAX)
        q = q + h * dq
    return q, dq


def gravity_comp(q):
    """Nominal-parameter gravity compensation (used by the exploration policy)."""
    m1, m2 = M1, M2
    g1 = (m1 * L1 / 2 + m2 * L1) * G * np.cos(q[:, 0]) + m2 * L2 / 2 * G * np.cos(q[:, 0] + q[:, 1])
    g2 = m2 * L2 / 2 * G * np.cos(q[:, 0] + q[:, 1])
    return np.stack([g1, g2], 1)


def rollout_logs(B, T, p, rng, sigma=1.5):
    """Simulate B exploration trajectories of T control steps.

    Returns q (B,T+1,2), dq (B,T+1,2), tau_cmd (B,T,2).
    """
    q = np.stack([rng.uniform(-np.pi, np.pi, B), rng.uniform(-2, 2, B)], 1)
    dq = rng.normal(0, 0.5, (B, 2))
    Q = np.zeros((B, T + 1, 2)); DQ = np.zeros((B, T + 1, 2)); TAU = np.zeros((B, T, 2))
    Q[:, 0], DQ[:, 0] = q, dq
    n = np.zeros((B, 2))
    for t in range(T):
        n = 0.8 * n + 0.6 * sigma * rng.standard_normal((B, 2))
        tau = np.clip(gravity_comp(q) + n, -TAU_MAX, TAU_MAX)
        TAU[:, t] = tau
        idx = np.clip(t - p['lat'], 0, None)
        applied = TAU[np.arange(B), idx]
        q, dq = step(q, dq, applied, p)
        Q[:, t + 1], DQ[:, t + 1] = q, dq
    return Q, DQ, TAU


def features(q, dq, tau_hist):
    """q,dq: (n,2); tau_hist: (n, NLAG+1, 2) ordered [t, t-1, ..., t-NLAG]."""
    return np.concatenate([np.sin(q), np.cos(q), dq, tau_hist.reshape(len(q), -1)], 1)


FEATURE_NAMES = (['sin q1', 'sin q2', 'cos q1', 'cos q2', 'dq1', 'dq2'] +
                 [f'tau{j}(t-{k})' if k else f'tau{j}(t)' for k in range(NLAG + 1) for j in (1, 2)])


def transitions(Q, DQ, TAU):
    """Build (X, Y, traj_id, t) from logs. Y = next state - state (q1,q2,dq1,dq2)."""
    B, T1, _ = Q.shape
    T = T1 - 1
    ts = np.arange(NLAG, T)
    bi, ti = np.meshgrid(np.arange(B), ts, indexing='ij')
    bi, ti = bi.ravel(), ti.ravel()
    hist = np.stack([TAU[bi, ti - k] for k in range(NLAG + 1)], 1)
    X = features(Q[bi, ti], DQ[bi, ti], hist)
    S = np.concatenate([Q, DQ], 2)
    Y = S[bi, ti + 1] - S[bi, ti]
    return X, Y, bi, ti


def sim_predict_delta(X, p_scalar, unmodeled=False, clip_lat=False):
    """Simulator used as a one-step model on feature rows (system identification).

    The feature rows hold torque history only up to t-NLAG. A delay longer than NLAG
    cannot be represented from these rows; this raises unless clip_lat=True, in which
    case the oldest available torque is used (a history-constrained reference, NOT exact).
    """
    n = len(X)
    q = np.arctan2(X[:, 0:2], X[:, 2:4])
    dq = X[:, 4:6]
    hist = X[:, 6:].reshape(n, NLAG + 1, 2)
    lat = int(p_scalar.get('lat', 0))
    if lat > NLAG and not clip_lat:
        raise ValueError(f'latency {lat} exceeds the {NLAG}-step torque history in the features')
    tau = hist[:, min(lat, NLAG)]
    p = params(n, m=p_scalar.get('m', 1.0), d=p_scalar.get('d', 1.0), f=p_scalar.get('f', 1.0), lat=0,
               unmodeled=unmodeled)
    q2, dq2 = step(q, dq, tau, p)
    return np.concatenate([q2 - q, dq2 - dq], 1)
