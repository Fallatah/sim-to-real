"""EDA, missingness diagnostics, feature selection, SHAP/ICE, ROC-AUC fault detection, PrefixSpan."""
import json, pickle, warnings
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=__import__('sklearn.exceptions', fromlist=['x']).ConvergenceWarning)  # diagnostics only; convergence is recorded in the main experiments
import sim, models, faults
from experiments import base, testset, data_seed, SIZES

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'font.family': 'serif',
                     'axes.spines.top': False, 'axes.spines.right': False})
S = {}
B = base()
rng = np.random.default_rng(99)

def fit_res(regime, axis, size, N, seed=0):
    tp = dict(sim.NOMINAL); tp[axis] = SIZES[axis][size]
    r = np.random.default_rng(data_seed(regime, axis, size, N, seed))
    R, TAU = faults.collect(N, tp, regime == 'out', r)
    sets, _ = faults.to_sets([R], TAU, N)
    return models.ResidualModel(B, sets, seed=seed), sets[0]

# ------------------------------------------------------------------ EDA (Topic 2)
src = np.load('out/source_sample.npy')
Xs, Ys = src[:, :16], src[:, 16:]
eda = {}
for i, n in enumerate(sim.FEATURE_NAMES[:6] + ['tau1(t)', 'tau2(t)']):
    k = i if i < 6 else 6 + (i - 6)
    x = Xs[:, k]
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    eda[n] = dict(mean=float(x.mean()), median=float(q2), iqr=float(q3 - q1), sd=float(x.std()), skew=float(stats.skew(x)))
for j, n in enumerate(['dq1_next-dq1', 'dq2_next-dq2']):
    x = Ys[:, 2 + j]
    eda[n] = dict(skew=float(stats.skew(x)), kurt=float(stats.kurtosis(x)))
S['eda'] = eda
# error distribution of a residual model (out, m small, N=1000)
res, _ = fit_res('out', 'm', 'small', 1000)
Q, DQ, TAU, Xt, Yt, sd = testset('m', 1.2, True)
_, err = models.multistep_nrmse(res, Q, DQ, TAU, state_sd=sd)
e = np.sqrt((err ** 2).mean(1))
q1, med, q3 = np.percentile(e, [25, 50, 75])
S['err_dist'] = dict(mean=float(e.mean()), median=float(med), iqr=float(q3 - q1), skew=float(stats.skew(e)),
                     tukey_out=float(np.mean(e > q3 + 1.5 * (q3 - q1))))
fig, axs = plt.subplots(1, 3, figsize=(7.16, 1.7))
axs[0].hist(Xs[:, 4], 60, color='#1f6fb4'); axs[0].set_xlabel('dq1 (rad/s), source'); axs[0].set_ylabel('count')
axs[1].boxplot([Xs[:, 4], Xs[:, 5], Xs[:, 6], Xs[:, 7]], tick_labels=['dq1', 'dq2', 'tau1', 'tau2'], vert=True,
               flierprops=dict(ms=1, alpha=0.3)); axs[1].set_title('Feature spread (Tukey whiskers)')
axs[2].hist(e, 60, color='#d1495b'); axs[2].axvline(e.mean(), color='k', lw=0.8, label='mean')
axs[2].axvline(med, color='k', ls='--', lw=0.8, label='median'); axs[2].legend(frameon=False)
axs[2].set_xlabel('per-start 10-step error (normalized)')
fig.tight_layout(pad=0.3); fig.savefig('fig/eda.pdf'); plt.close(fig)

# ------------------------------------------------------------------ missingness diagnostic (Topic 3)
tp = dict(sim.NOMINAL); tp['m'] = 1.2
R, TAU_ = faults.collect(3000, tp, True, np.random.default_rng(5))
diag = {}
for mech in ['MCAR', 'MAR', 'MNAR']:
    M = faults.missing_mask(R, TAU_, mech, 0.3, np.random.default_rng(6))
    tau = np.concatenate([TAU_, TAU_[:, -1:]], 1)
    out = {}
    m0 = M[..., 0].ravel()
    # observed covariates only: |tau1| (always observed) and |dq2| when joint 2 observed
    a = np.abs(tau[..., 0]).ravel()
    out['tau1'] = stats.ttest_ind(a[m0], a[~m0], equal_var=False).pvalue
    ok2 = ~M[..., 1].ravel()
    b = np.abs(R[..., 3]).ravel()
    out['dq2'] = stats.ttest_ind(b[m0 & ok2], b[~m0 & ok2], equal_var=False).pvalue
    out['mean_dq2_miss'] = float(b[m0 & ok2].mean()); out['mean_dq2_obs'] = float(b[~m0 & ok2].mean())
    out['mean_tau1_miss'] = float(a[m0].mean()); out['mean_tau1_obs'] = float(a[~m0].mean())
    diag[mech] = out
S['mech_ttest'] = diag

# why tabular imputation fails: imputed readings corrupt the state-change targets
Rc, Tc = faults.collect(1000, tp, True, np.random.default_rng(7))
Xc0, Yc0, _, _ = sim.transitions(Rc[..., :2], Rc[..., 2:], Tc)
Mm = faults.missing_mask(Rc, Tc, 'MCAR', 0.1, np.random.default_rng(8))
tgt = {}
for strat in ['median', 'mice', 'interp', 'lagreg']:
    Ri, _ = faults.impute(faults.apply_mask(Rc, Mm), Tc, strat, 0)
    Xi, Yi, bi_, ti_ = sim.transitions(Ri[0][..., :2], Ri[0][..., 2:], Tc)
    touched = (Mm[bi_, ti_] | Mm[bi_, ti_ + 1]).any(1)
    err = np.abs((Yi - Yc0) / B.sc.ysd)
    tgt[strat] = dict(frac_touched=float(touched.mean()),
                      target_err_touched=float(np.median(err[touched].max(1))),
                      share_sq_err_from_touched=float((err[touched] ** 2).sum() / (err ** 2).sum()))
S['imputed_targets'] = tgt

# ------------------------------------------------------------------ feature selection (Topic 3)
C = np.corrcoef(Xs.T)
pairs = [(sim.FEATURE_NAMES[i], sim.FEATURE_NAMES[j], float(C[i, j])) for i in range(16) for j in range(i + 1, 16)
         if abs(C[i, j]) > 0.8]
pca = PCA().fit(B.sc.sx(Xs))
cv = np.cumsum(pca.explained_variance_ratio_)
S['corr_pairs'] = pairs
S['pca'] = dict(k90=int(np.searchsorted(cv, 0.90) + 1), k95=int(np.searchsorted(cv, 0.95) + 1),
                first3=[float(v) for v in pca.explained_variance_ratio_[:3]])

# permutation importance (feature shuffling) on the adapted model, in-class mass vs latency
def perm_importance(model, X, Y, reps=5):
    base_e = np.mean(((model.predict(X) - Y) / B.sc.ysd) ** 2)
    imp = []
    for k in range(X.shape[1]):
        v = []
        for r in range(reps):
            Xp = X.copy(); Xp[:, k] = rng.permutation(Xp[:, k])
            v.append(np.mean(((model.predict(Xp) - Y) / B.sc.ysd) ** 2) - base_e)
        imp.append(np.mean(v))
    return np.array(imp)

# ------------------------------------------------------------------ SHAP (permutation, antithetic) + ICE (Topic 7)
def perm_shap(f, X, bg, n_perm=10):
    n, d = X.shape
    phi = np.zeros((n, d))
    for i in range(n):
        for _ in range(n_perm):
            order = rng.permutation(d)
            for o in (order, order[::-1]):
                z = bg[rng.integers(0, len(bg))].copy()
                prev = f(z[None])[0]
                for k in o:
                    z[k] = X[i, k]
                    cur = f(z[None])[0]
                    phi[i, k] += cur - prev
                    prev = cur
        phi[i] /= 2 * n_perm
    return phi

shap_res, perm_res = {}, {}
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.0), gridspec_kw=dict(width_ratios=[1, 1, 1]))
for a, (axis, title) in zip(axs[:2], [('m', 'link mass +60%'), ('lat', 'latency +4 steps')]):
    model, (Xa, Ya) = fit_res('in', axis, 'large', 3000)
    Qx, DQx, TAUx, Xt2, Yt2, _ = testset(axis, SIZES[axis]['large'], False)
    Xsc = B.sc.sx(Xt2)
    f = lambda Z: model.residual_only(Z)[:, 2]          # residual correction to delta dq1
    idx = rng.choice(len(Xsc), 60, replace=False)
    phi = perm_shap(f, Xsc[idx], Xsc[rng.choice(len(Xsc), 50, replace=False)])
    mabs = np.abs(phi).mean(0)
    shap_res[axis] = {sim.FEATURE_NAMES[k]: float(mabs[k]) for k in range(16)}
    # additivity check
    pi = perm_importance(model, Xt2[:2000], Yt2[:2000])
    perm_res[axis] = {sim.FEATURE_NAMES[k]: float(pi[k]) for k in range(16)}
    o = np.argsort(mabs)[::-1][:8][::-1]
    a.barh(range(8), mabs[o], color='#1f6fb4'); a.set_yticks(range(8)); a.set_yticklabels([sim.FEATURE_NAMES[k] for k in o], fontsize=6)
    a.set_title(f'mean |SHAP|, {title}'); a.set_xlabel('effect on residual $\\Delta\\dot q_1$ (scaled)')
    if axis == 'm':
        # centered ICE for tau1(t)
        k = 6
        grid = np.linspace(np.percentile(Xt2[:, k], 5), np.percentile(Xt2[:, k], 95), 25)
        sel = rng.choice(len(Xt2), 40, replace=False)
        cols = plt.cm.viridis((Xt2[sel, 2] + 1) / 2)
        curves = []
        for ii, r_ in enumerate(sel):
            Z = np.repeat(Xt2[r_:r_ + 1], len(grid), 0); Z[:, k] = grid
            y = model.residual_only(B.sc.sx(Z))[:, 2]
            y = y - y[0]; curves.append(y)
            axs[2].plot(grid, y, color=cols[ii], lw=0.6, alpha=0.8)
        axs[2].plot(grid, np.mean(curves, 0), 'k', lw=1.6, label='mean (PDP)')
        axs[2].set_xlabel('tau1(t) (N m)'); axs[2].set_ylabel('centered residual')
        axs[2].set_title('Centered ICE, link mass +60%'); axs[2].legend(frameon=False)
        S['ice_slope_sd'] = float(np.std([np.polyfit(grid, c, 1)[0] for c in curves]))
        S['ice_slope_mean'] = float(np.mean([np.polyfit(grid, c, 1)[0] for c in curves]))
fig.tight_layout(pad=0.3); fig.savefig('fig/shap.pdf'); plt.close(fig)
S['shap'] = shap_res; S['perm_importance'] = perm_res

# ------------------------------------------------------------------ ensemble spread as fault detector (Topic 6)
Xc = Xt.copy()
hit = rng.random(len(Xc)) < 0.5
sgn = rng.choice([-1, 1], (len(Xc), 2))
Xc[hit, 4:6] += sgn[hit] * 5 * Xt[:, 4:6].std(0)
P = B.predict_members(Xc)
spread = (P.std(0) / B.sc.ysd).mean(1)
S['auc_spread'] = float(roc_auc_score(hit, spread))
z = np.abs((Xc[:, 4:6] - np.median(Xt[:, 4:6], 0)) / (1.4826 * np.median(np.abs(Xt[:, 4:6] - np.median(Xt[:, 4:6], 0)), 0))).max(1)
S['auc_madz'] = float(roc_auc_score(hit, z))

# ------------------------------------------------------------------ PrefixSpan on failure windows (Topic 5)
def prefixspan(db, minsup, maxlen=3):
    """db: list of sequences of itemsets (frozensets). Patterns of single-item elements."""
    out = {}
    def project(seqs, item):
        proj = []
        for s, start in seqs:
            for i in range(start, len(s)):
                if item in s[i]:
                    proj.append((s, i + 1)); break
        return proj
    def grow(prefix, seqs):
        if len(prefix) >= maxlen:
            return
        counts = {}
        for s, start in seqs:
            seen = set()
            for el in s[start:]:
                seen |= el
            for it in seen:
                counts[it] = counts.get(it, 0) + 1
        for it, c in counts.items():
            if c >= minsup:
                p = prefix + (it,)
                out[p] = c
                grow(p, project(seqs, it))
    grow((), [(s, 0) for s in db])
    return out

model = res                                            # out, m small, N=1000
frng = np.random.default_rng(31)
Qf, DQf = Q.copy(), DQ.copy()
Bn, T1, _ = Qf.shape
drop = frng.random((Bn, T1, 2)) < 0.05
spike = frng.random((Bn, T1, 2)) < 0.02
DQo = DQf + spike * frng.choice([-1, 1], (Bn, T1, 2)) * 5 * DQ.reshape(-1, 2).std(0)
Qo = Qf.copy()
for t in range(1, T1):                                  # dropout: hold last reading
    Qo[:, t][drop[:, t]] = Qo[:, t - 1][drop[:, t]]
    DQo[:, t][drop[:, t]] = DQo[:, t - 1][drop[:, t]]
ts = np.arange(sim.NLAG, T1 - 1)
bi, ti = np.meshgrid(np.arange(Bn), ts, indexing='ij'); bi, ti = bi.ravel(), ti.ravel()
hist = np.stack([TAU[bi, ti - k] for k in range(sim.NLAG + 1)], 1)
Xo = sim.features(Qo[bi, ti], DQo[bi, ti], hist)
Strue = np.concatenate([Q, DQ], 2)
Ytrue = Strue[bi, ti + 1] - Strue[bi, ti]
Xcl = sim.features(Q[bi, ti], DQ[bi, ti], hist)
e_clean = np.sqrt((((model.predict(Xcl) - Ytrue) / B.sc.ysd) ** 2).mean(1))
e_obs = np.sqrt((((model.predict(Xo) - Ytrue) / B.sc.ysd) ** 2).mean(1)).reshape(Bn, -1)
thr = np.percentile(e_clean, 99)
vthr = np.percentile(np.abs(DQ), 90); tthr = np.percentile(np.abs(TAU), 90)
def events(b, t):
    s = set()
    if drop[b, t].any(): s.add('D')
    if spike[b, t].any(): s.add('S')
    if (np.abs(DQ[b, t]) > vthr).any(): s.add('V')
    if (np.abs(TAU[b, t]) > tthr).any(): s.add('T')
    if t > 0 and (np.sign(TAU[b, t]) != np.sign(TAU[b, t - 1])).any(): s.add('R')
    return frozenset(s)
fail_db, ok_db = [], []
for b in range(Bn):
    for k, t in enumerate(ts):
        if t < 4:
            continue
        seq = [events(b, u) for u in range(t - 4, t + 1)]
        (fail_db if e_obs[b, k] > thr else ok_db).append(seq)
pf = prefixspan(fail_db, max(2, int(0.2 * len(fail_db))))
po = prefixspan(ok_db, 1)
rows = []
for p, c in pf.items():
    sup_f = c / len(fail_db); sup_o = po.get(p, 0) / len(ok_db)
    rows.append(dict(pattern='<' + ' '.join(p) + '>', sup_fail=sup_f, sup_ok=sup_o,
                     lift=sup_f / sup_o if sup_o > 0 else float('inf')))
rows.sort(key=lambda r: (-r['lift'], -r['sup_fail']))
S['prefixspan'] = dict(n_fail=len(fail_db), n_ok=len(ok_db), top=rows[:8],
                       fail_rate=float(len(fail_db) / (len(fail_db) + len(ok_db))))
json.dump(S, open('out/extras.json', 'w'), indent=1, default=float)
print('extras done')
