"""Random-shooting MPC reaching task, using any one-step model."""
import numpy as np
import sim

T_EP, H, K = 60, 10, 256
TOL_Q, TOL_DQ = 0.15, 1.0


def episode(model, p_true, q0, dq0, qstar, rng):
    """Returns success (bool): final 5 steps within tolerance."""
    q, dq = q0[None].copy(), dq0[None].copy()
    hist = np.zeros((sim.NLAG + 1 + 10, 2))          # past commands, most recent first
    hist[:] = sim.gravity_comp(q)[0]
    applied_buf = [hist[0].copy() for _ in range(10)]
    ok = []
    for t in range(T_EP):
        # sample K torque offsets, piecewise constant over horizon with small jitter
        u0 = rng.normal(0, 2.5, (K, 1, 2)) + 0.5 * rng.normal(0, 1, (K, H, 2))
        qq = np.repeat(q, K, 0); dd = np.repeat(dq, K, 0)
        hk = np.repeat(hist[None, :sim.NLAG + 1], K, 0)          # (K, NLAG+1, 2)
        cost = np.zeros(K)
        for h in range(H):
            tau = np.clip(sim.gravity_comp(qq) + u0[:, h], -sim.TAU_MAX, sim.TAU_MAX)
            hk = np.concatenate([tau[:, None], hk[:, :-1]], 1)
            d = model.predict(sim.features(qq, dd, hk))
            qq, dd = qq + d[:, :2], np.clip(dd + d[:, 2:], -sim.DQ_MAX, sim.DQ_MAX)
            w = 1.0 if h < H - 1 else 5.0
            cost += w * ((qq - qstar) ** 2).sum(1) + 0.01 * w * (dd ** 2).sum(1)
        best = np.argmin(cost)
        tau = np.clip(sim.gravity_comp(q)[0] + u0[best, 0], -sim.TAU_MAX, sim.TAU_MAX)
        hist = np.concatenate([tau[None], hist[:-1]], 0)
        lat = int(p_true['lat'][0])
        applied = hist[min(lat, len(hist) - 1)]
        q, dq = sim.step(q, dq, applied[None], p_true)
        ok.append(np.linalg.norm(q[0] - qstar) < TOL_Q and np.linalg.norm(dq[0]) < TOL_DQ)
    return all(ok[-5:])


def tasks(n, rng):
    q0 = np.stack([-np.pi / 2 + rng.uniform(-0.3, 0.3, n), rng.uniform(-0.3, 0.3, n)], 1)
    dq0 = rng.normal(0, 0.2, (n, 2))
    qs = np.stack([rng.uniform(-1.2, 0.3, n), rng.uniform(-1.0, 1.0, n)], 1)
    return q0, dq0, qs


def success_rate(model, p_scalar, unmodeled, n, seed):
    rng = np.random.default_rng(seed)
    q0, dq0, qs = tasks(n, rng)
    p = sim.params(1, **p_scalar, unmodeled=unmodeled)
    return float(np.mean([episode(model, p, q0[i], dq0[i], qs[i], rng) for i in range(n)]))
