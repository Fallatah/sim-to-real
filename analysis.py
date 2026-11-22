"""Statistics and figures from out/*.csv. Writes out/stats.json and fig/*.pdf."""
import json, os, warnings
import numpy as np, pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT, FIG = 'out', 'fig'
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'font.family': 'serif',
                     'axes.spines.top': False, 'axes.spines.right': False})
COL = {'Residual': '#1f6fb4', 'SysID': '#d1495b', 'DR': '#6c757d', 'Oracle': '#2a9d55'}
AXN = {'m': 'link mass', 'd': 'joint damping', 'f': 'friction', 'lat': 'actuator latency'}
RNG = np.random.default_rng(0)
S = {}


def boot(x, B=4000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = RNG.choice(x, (B, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p, dtype=float)
    run = 0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k]); adj[k] = min(1, run)
    return adj


E1 = pd.read_csv(f'{OUT}/E1.csv'); DR = pd.read_csv(f'{OUT}/DR.csv')
BUD = sorted(E1.N.unique())

# ------------------------------------------------------------------ E1 scaling
def plaw(N, a, b, c):
    return a * N ** (-b) + c

pl = []
for (reg, ax, sz), g in E1[E1.method == 'Residual'].groupby(['regime', 'axis', 'size']):
    piv = g.pivot(index='seed', columns='N', values='nrmse10')[BUD]
    def fit(M):
        try:
            p, _ = optimize.curve_fit(plaw, np.array(BUD, float), M, p0=[1, 0.5, 0.05],
                                      bounds=([0, 0, 0], [100, 3, 2]), maxfev=20000)
            return p
        except Exception:
            return [np.nan] * 3
    p0 = fit(piv.mean().values)
    yv = piv.mean().values
    r2fit = 1 - np.sum((yv - plaw(np.array(BUD, float), *p0)) ** 2) / np.sum((yv - yv.mean()) ** 2)
    yv3 = yv[2:]; N3 = np.array(BUD[2:], float)
    p3 = fit3 = None
    try:
        p3, _ = optimize.curve_fit(plaw, N3, yv3, p0=[1, 0.5, 0.05], bounds=([0, 0, 0], [100, 3, 2]), maxfev=20000)
        r2_100 = 1 - np.sum((yv3 - plaw(N3, *p3)) ** 2) / np.sum((yv3 - yv3.mean()) ** 2)
    except Exception:
        p3, r2_100 = [np.nan] * 3, np.nan
    mono = bool(np.all(np.diff(yv) <= 0))
    bs = np.array([fit(piv.iloc[RNG.integers(0, len(piv), len(piv))].mean().values) for _ in range(300)])
    pl.append(dict(regime=reg, axis=ax, size=sz, r2=r2fit, r2_100=r2_100, b100=p3[1], c100=p3[2], monotone=mono,
                   b=p0[1], b_lo=np.nanpercentile(bs[:, 1], 2.5),
                   b_hi=np.nanpercentile(bs[:, 1], 97.5), c=p0[2], c_lo=np.nanpercentile(bs[:, 2], 2.5),
                   c_hi=np.nanpercentile(bs[:, 2], 97.5)))
S['powerlaw'] = pl

# paired residual vs sysid per N (out-of-class), Holm across all (axis,size,N)
tests = []
for (reg, ax, sz, N), g in E1.groupby(['regime', 'axis', 'size', 'N']):
    r = g[g.method == 'Residual'].sort_values('seed').nrmse10.values
    s = g[g.method == 'SysID'].sort_values('seed').nrmse10.values
    d = DR[(DR.regime == reg) & (DR.axis == ax) & (DR['size'] == sz) & (DR.method == 'DR')].nrmse10.iloc[0]
    diff = r - s
    t, p = stats.ttest_rel(r, s) if np.std(diff) > 0 else (np.nan, 1.0)
    # check on the paired differences: normality, and a rank-based test that does not assume it
    sh_p = stats.shapiro(diff).pvalue if np.std(diff) > 0 else np.nan
    wx_p = stats.wilcoxon(r, s).pvalue if np.any(diff != 0) else 1.0
    # DR enters at its actual (constant) value: no artificial perturbation
    try:
        fr = stats.friedmanchisquare(r, s, np.full_like(r, d)).pvalue
    except Exception:
        fr = np.nan
    tests.append(dict(regime=reg, axis=ax, size=sz, N=N, res=r.mean(), sys=s.mean(), dr=d, t=t, p=p,
                      shapiro_p=sh_p, wilcoxon_p=wx_p, friedman_p=fr, diff_mean=diff.mean(),
                      diff_min=diff.min(), diff_max=diff.max()))
T = pd.DataFrame(tests)
# Correction family: all (mismatch type x size x budget) residual-vs-SysID comparisons within
# one regime (56 tests per regime); Holm applied separately to t-test and Wilcoxon p-values.
for reg in ['in', 'out']:
    k = T.regime == reg
    T.loc[k, 'p_holm'] = holm(T.loc[k, 'p'].values)
    T.loc[k, 'wilcoxon_holm'] = holm(T.loc[k, 'wilcoxon_p'].values)
    T.loc[k, 'friedman_holm'] = holm(T.loc[k, 'friedman_p'].fillna(1).values)
T.to_csv(f'{OUT}/E1_tests.csv', index=False)

# crossover N (out-of-class): first budget from which residual mean stays below sysid
def crossover(df):
    piv_r = df[df.method == 'Residual'].pivot(index='seed', columns='N', values='nrmse10')[BUD]
    piv_s = df[df.method == 'SysID'].pivot(index='seed', columns='N', values='nrmse10')[BUD]
    def cx(idx):
        d = (piv_r.iloc[idx].mean() - piv_s.iloc[idx].mean()).values
        lN = np.log10(BUD)
        for i in range(len(d) - 1):
            if d[i] > 0 and (d[i + 1:] <= 0).all():
                return 10 ** (lN[i] + (lN[i + 1] - lN[i]) * d[i] / (d[i] - d[i + 1]))
        return np.nan if d[-1] > 0 else BUD[0]
    c0 = cx(np.arange(len(piv_r)))
    bs = [cx(RNG.integers(0, len(piv_r), len(piv_r))) for _ in range(1000)]
    return c0, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), float(np.mean(np.isnan(bs)))

cxs = []
for (ax, sz), g in E1[E1.regime == 'out'].groupby(['axis', 'size']):
    c, lo, hi, never = crossover(g)
    cxs.append(dict(axis=ax, size=sz, cross=c, lo=lo, hi=hi, frac_never=never))
S['crossover_out'] = cxs

# residual vs DR: smallest N at which residual beats DR (mean)
beats = []
for (reg, ax, sz), g in T.groupby(['regime', 'axis', 'size']):
    g = g.sort_values('N')
    ok = g[g.res < g.dr].N.values
    beats.append(dict(regime=reg, axis=ax, size=sz, N_beat_DR=int(ok.min()) if len(ok) else None,
                      res_N10=float(g.res.iloc[0]), dr=float(g.dr.iloc[0])))
S['res_vs_dr'] = beats

# summary table at selected budgets
tab = E1.groupby(['regime', 'axis', 'size', 'N', 'method']).nrmse10.apply(lambda x: boot(x)).reset_index()
tab[['mean', 'lo', 'hi']] = pd.DataFrame(tab.nrmse10.tolist(), index=tab.index)
tab.drop(columns='nrmse10').to_csv(f'{OUT}/E1_summary.csv', index=False)

# one-step metrics at N=1000/10000 (out, m, large)
os1 = E1[(E1.regime == 'out') & (E1.axis == 'm') & (E1['size'] == 'large') & (E1.N.isin([1000, 10000]))]
S['onestep_out_m_large'] = os1.groupby(['N', 'method'])[['rmse1', 'mae1', 'r2']].mean().reset_index().to_dict('records')
S['onestep_DR'] = DR[(DR.regime == 'out') & (DR.axis == 'm') & (DR['size'] == 'large')][['method', 'rmse1', 'mae1', 'r2']].to_dict('records')

# test-time shift ratio at N=1000
sh = E1[E1.N == 1000].copy(); sh['ratio'] = sh.nrmse10_shift / sh.nrmse10
S['shift'] = (sh[sh.method == 'Residual'].groupby(['regime', 'axis', 'size']).ratio.mean().reset_index().to_dict('records'))
S['shift_abs'] = sh.groupby(['regime', 'method', 'axis', 'size'])[['nrmse10', 'nrmse10_shift']].mean().reset_index().to_dict('records')
drs = DR[DR.method == 'DR'].copy(); drs['ratio'] = drs.nrmse10_shift / drs.nrmse10
S['shift_DR'] = drs[['regime', 'axis', 'size', 'nrmse10', 'nrmse10_shift', 'ratio']].to_dict('records')

# Figure: scaling curves (large mismatch), rows = regime
fig, axs = plt.subplots(2, 4, figsize=(7.16, 3.3), sharex=True)
for i, reg in enumerate(['in', 'out']):
    for j, ax in enumerate(['m', 'd', 'f', 'lat']):
        a = axs[i, j]
        for meth in ['Residual', 'SysID']:
            g = tab[(tab.regime == reg) & (tab.axis == ax) & (tab['size'] == 'large') & (tab.method == meth)].sort_values('N')
            a.plot(g.N, g['mean'], 'o-', ms=2.5, lw=1.2, color=COL[meth], label=meth)
            a.fill_between(g.N, g.lo, g.hi, color=COL[meth], alpha=0.2, lw=0)
        for meth, ls in [('DR', '--'), ('Oracle', ':')]:
            v = DR[(DR.regime == reg) & (DR.axis == ax) & (DR['size'] == 'large') & (DR.method == meth)].nrmse10.iloc[0]
            a.axhline(v, color=COL[meth], ls=ls, lw=1, label=meth if meth == 'DR' else 'True simulator')
        a.set_xscale('log'); a.set_ylim(bottom=0)
        if i == 0:
            a.set_title(AXN[ax] + (' (+4 steps)' if ax == 'lat' else ' (+60%)'))
        if j == 0:
            a.set_ylabel(('In-class' if reg == 'in' else 'Out-of-class') + '\n10-step nRMSE')
        if i == 1:
            a.set_xlabel('target transitions $N$')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94))
fig.savefig(f'{FIG}/scaling.pdf'); plt.close(fig)

# ------------------------------------------------------------------ robustness ratios
clean = E1[(E1.regime == 'out') & (E1.N == 1000)][['axis', 'size', 'seed', 'method', 'nrmse10']].rename(columns={'nrmse10': 'clean'})
clean_in = E1[(E1.regime == 'in') & (E1.N == 1000)][['axis', 'size', 'seed', 'method', 'nrmse10']].rename(columns={'nrmse10': 'clean'})

def ratios(df, reg='out'):
    c = clean if reg == 'out' else clean_in
    df = df.merge(c, on=['axis', 'size', 'seed', 'method'])
    df['ratio'] = df.nrmse10 / df.clean
    return df

E2 = pd.concat([pd.read_csv(f'{OUT}/E2.csv'), pd.read_csv(f'{OUT}/E2b.csv')])
E2o = ratios(E2[E2.regime == 'out'])
r2 = []
for (mech, rate, imp, meth), g in E2o.groupby(['mech', 'rate', 'imputer', 'method']):
    m, lo, hi = boot(g.ratio)
    r2.append(dict(mech=mech, rate=rate, imputer=imp, method=meth, ratio=m, lo=lo, hi=hi, robust=bool(hi < 1.25),
                   abs=float(g.nrmse10.mean()), n_used=float(g.n_used.mean())))
R2 = pd.DataFrame(r2); R2.to_csv(f'{OUT}/E2_ratios.csv', index=False)
PRE = ['listwise', 'median', 'median+ind', 'knn', 'regression', 'mice', 'interp']
R2p = R2[R2.imputer.isin(PRE)]
S['E2_robust_count'] = R2p.groupby('method').robust.sum().to_dict()
S['E2_cells'] = int(len(R2p) / 2)
# sensitivity of the robust counts to the 1.25 threshold
S['E2_threshold_sensitivity'] = {str(th): (R2p.hi < th).groupby(R2p.method).sum().to_dict() for th in [1.1, 1.25, 1.5, 2.0]}
# listwise deletion vs a clean dataset of the same size (selection bias vs fewer examples)
lm = []
for (mech, rate, meth), g in E2o[E2o.imputer.isin(['listwise', 'clean_matched'])].groupby(['mech', 'rate', 'method']):
    piv = g.pivot(index='seed', columns='imputer', values='nrmse10').dropna()
    d = piv['listwise'] - piv['clean_matched']
    lm.append(dict(mech=mech, rate=rate, method=meth, listwise=float(piv['listwise'].mean()),
                   matched=float(piv['clean_matched'].mean()), diff=float(d.mean()),
                   p=float(stats.wilcoxon(piv['listwise'], piv['clean_matched']).pvalue) if (d != 0).any() else 1.0,
                   n_used=float(g[g.imputer == 'listwise'].n_used.mean())))
S['listwise_vs_matched'] = lm
S['E2_imp_warnings'] = E2.groupby('imputer').imp_warnings.mean().to_dict()
S['res_converged'] = float(pd.concat([E1, E2])[lambda d: d.method == 'Residual'].res_converged.mean())
S['E2_in'] = E2[E2.regime == 'in'].groupby(['mech', 'imputer', 'method']).nrmse10.mean().reset_index().to_dict('records')
S['clean_ref'] = {m: boot(clean[(clean.axis == 'm') & (clean['size'] == 'small') & (clean.method == m)].clean)
                  for m in ['Residual', 'SysID']}
S['DR_ref'] = float(DR[(DR.regime == 'out') & (DR.axis == 'm') & (DR['size'] == 'small') & (DR.method == 'DR')].nrmse10.iloc[0])
# mechanism effect: Friedman across mechanisms at 30%, per imputer (residual), paired by seed
mfx = []
for imp, g in E2o[(E2o.rate == 0.3) & (E2o.method == 'Residual')].groupby('imputer'):
    piv = g.pivot(index='seed', columns='mech', values='nrmse10')
    p = stats.friedmanchisquare(*[piv[c] for c in piv.columns]).pvalue
    mfx.append(dict(imputer=imp, **piv.mean().to_dict(), p=p))
S['mech_effect_res'] = mfx

IMP = ['listwise', 'median', 'median+ind', 'knn', 'regression', 'mice', 'interp', 'lagreg', 'clean_matched']
IML = ['Listwise', 'Median', 'Median+ind.', 'KNN', 'Regression', 'MICE', 'Interp.$^\\dagger$', 'Lag-aware$^\\ddagger$',
       'Matched clean$^\\ddagger$']
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw=dict(wspace=0.45))
cols = [(m, r) for m in ['MCAR', 'MAR', 'MNAR'] for r in [0.1, 0.3, 0.5]]
for a, meth in zip(axs, ['Residual', 'SysID']):
    M = np.array([[R2[(R2.method == meth) & (R2.imputer == i) & (R2.mech == m) & (R2.rate == r)].ratio.iloc[0]
                   for (m, r) in cols] for i in IMP])
    rb = np.array([[R2[(R2.method == meth) & (R2.imputer == i) & (R2.mech == m) & (R2.rate == r)].robust.iloc[0]
                    for (m, r) in cols] for i in IMP])
    im = a.imshow(np.log2(M), cmap='RdYlBu_r', vmin=-3, vmax=3, aspect='auto')
    for y in range(M.shape[0]):
        for x in range(M.shape[1]):
            a.text(x, y, f'{M[y, x]:.1f}' + ('*' if rb[y, x] else ''), ha='center', va='center', fontsize=5.5,
                   color='white' if abs(np.log2(M[y, x])) > 2 else 'black')
    a.set_xticks(range(9)); a.set_xticklabels([f'{int(r*100)}%' for m, r in cols], fontsize=6)
    for gx, mname in zip((1, 4, 7), ['MCAR', 'MAR', 'MNAR']):
        a.text(gx, -0.65, mname, ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    a.set_yticks(range(len(IMP))); a.set_yticklabels(IML, fontsize=6)
    a.set_title(f'{meth}: error ratio vs. clean data', pad=12)
    for x in (2.5, 5.5):
        a.axvline(x, color='white', lw=1.5)
cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01)
cb.set_ticks([-3, -1, 0, 1, 3]); cb.set_ticklabels(['1/8', '1/2', '1', '2', '8']); cb.ax.tick_params(labelsize=6)
fig.savefig(f'{FIG}/missing.pdf', bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------------ E3 outliers/noise
E3 = ratios(pd.read_csv(f'{OUT}/E3.csv'))
r3 = []
for (o, tr, nz, meth), g in E3.groupby(['outl', 'treat', 'noise', 'method']):
    m, lo, hi = boot(g.ratio)
    r3.append(dict(outl=o, treat=tr, noise=nz, method=meth, ratio=m, lo=lo, hi=hi, robust=bool(hi < 1.25),
                   abs=float(g.nrmse10.mean()), n_used=float(g.n_used.mean()), param=float(g.param.mean())))
R3 = pd.DataFrame(r3); R3.to_csv(f'{OUT}/E3_ratios.csv', index=False)
# H3 test: raw spikes, residual ratio vs sysid ratio (paired) at each rate
h3 = []
for o in [0.01, 0.05, 0.10]:
    g = E3[(E3.outl == o) & (E3.treat == 'raw')]
    r = g[g.method == 'Residual'].sort_values('seed').ratio.values
    s = g[g.method == 'SysID'].sort_values('seed').ratio.values
    h3.append(dict(outl=o, res=r.mean(), sys=s.mean(), p=stats.ttest_rel(r, s).pvalue))
for i, p in enumerate(holm([x['p'] for x in h3])):
    h3[i]['p_holm'] = p
S['H3'] = h3
S['E3_threshold_sensitivity'] = {str(th): (R3.hi < th).groupby(R3.method).sum().to_dict() for th in [1.1, 1.25, 1.5, 2.0]}

fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.2), sharey=True)
for a, meth in zip(axs, ['Residual', 'SysID']):
    for k, tr in enumerate(['raw', 'winsorize', 'discard']):
        g = R3[(R3.method == meth) & (R3.treat == tr) & (R3.noise == 0)].sort_values('outl')
        if tr == 'raw':
            g = g[g.outl > 0]
            x = np.arange(1, len(g) + 1) + (k - 1) * 0.27
        else:
            x = np.arange(len(g)) + (k - 1) * 0.27
        a.bar(x, g.ratio, 0.27, yerr=[g.ratio - g.lo, g.hi - g.ratio], color=['#adb5bd', '#e9a03b', '#7b5ea7'][k],
              label=tr.capitalize(), error_kw=dict(lw=0.6, capsize=1.2))
    a.set_xticks(range(4)); a.set_xticklabels(['0%', '1%', '5%', '10%']); a.axhline(1.25, color='k', lw=0.6, ls='--')
    a.set_title(meth); a.set_xlabel('velocity spikes')
axs[0].set_ylabel('error ratio vs. clean')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=6)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9)); fig.savefig(f'{FIG}/outliers.pdf'); plt.close(fig)

# ------------------------------------------------------------------ E6 budget x mechanism
E6 = pd.read_csv(f'{OUT}/E6.csv')
ref = E1[(E1.regime == 'out') & (E1.axis == 'm') & (E1['size'] == 'small')]
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
e6 = []
for a, meth in zip(axs, ['Residual', 'SysID']):
    g = ref[ref.method == meth].groupby('N').nrmse10.mean()
    a.plot(g.index, g.values, 'k-', lw=1.2, label='clean')
    for mech, c in zip(['MCAR', 'MAR', 'MNAR'], ['#1f6fb4', '#e9a03b', '#d1495b']):
        h = E6[(E6.method == meth) & (E6.mech == mech)].groupby('N').nrmse10.mean()
        a.plot(h.index, h.values, 'o-', ms=2, lw=1, color=c, label=mech)
        for N in BUD:
            e6.append(dict(method=meth, mech=mech, N=N, err=float(h.get(N, np.nan)), clean=float(g.get(N, np.nan))))
    a.set_xscale('log'); a.set_title(meth); a.set_xlabel('$N$')
axs[0].set_ylabel('10-step nRMSE')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=6)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9)); fig.savefig(f'{FIG}/cross.pdf'); plt.close(fig)
S['E6'] = e6

# ------------------------------------------------------------------ E7 by mismatch type (RQ-d)
E7 = ratios(pd.read_csv(f'{OUT}/E7.csv'))
E7['fault'] = np.where(E7.rate > 0, 'MAR30', 'spike5')
sc = E1[(E1.regime == 'out') & (E1['size'] == 'small') & (E1.N.isin([100, 1000]))]
sc = sc.pivot_table(index=['axis', 'seed', 'method'], columns='N', values='nrmse10').reset_index()
sc['ratio'] = sc[100] / sc[1000]; sc['fault'] = 'scarce100'
E7a = pd.concat([E7[['axis', 'seed', 'method', 'fault', 'ratio']], sc[['axis', 'seed', 'method', 'fault', 'ratio']]])
r7, kw = [], []
for (f, meth), g in E7a.groupby(['fault', 'method']):
    for ax, h in g.groupby('axis'):
        m, lo, hi = boot(h.ratio); r7.append(dict(fault=f, method=meth, axis=ax, ratio=m, lo=lo, hi=hi))
    kw.append(dict(fault=f, method=meth, p=stats.kruskal(*[h.ratio.values for _, h in g.groupby('axis')]).pvalue))
S['E7'] = r7; S['E7_kruskal'] = kw

# ------------------------------------------------------------------ E8 task success (two conditions)
E8 = pd.read_csv(f'{OUT}/E8.csv')
S['E8'] = {}
fig, axs8 = plt.subplots(1, 2, figsize=(3.5, 2.2), sharey=True)
for a, ax in zip(axs8, ['m', 'lat']):
    e8 = E8[E8.axis == ax]
    orc = e8[e8.method == 'Oracle'].success.mean(); drv = float(e8[e8.method == 'DR'].success.mean())
    r8 = []
    for (N, meth), g in e8[e8.method.isin(['Residual', 'SysID'])].groupby(['N', 'method']):
        m, lo, hi = boot(g.success); r8.append(dict(N=N, method=meth, success=m, lo=lo, hi=hi))
    R8 = pd.DataFrame(r8)
    thr = {meth: (int(R8[(R8.method == meth) & (R8.success >= 0.8 * orc)].N.min())
                  if (R8[R8.method == meth].success >= 0.8 * orc).any() else None) for meth in ['Residual', 'SysID']}
    sz = 'large'
    e1m = E1[(E1.regime == 'out') & (E1.axis == ax) & (E1['size'] == sz)].groupby(['N', 'method']).nrmse10.mean()
    drerr = float(DR[(DR.regime == 'out') & (DR.axis == ax) & (DR['size'] == sz) & (DR.method == 'DR')].nrmse10.iloc[0])
    S['E8'][ax] = dict(rows=r8, oracle=float(orc), DR=drv, threshold=thr, DR_err=drerr,
                       err={f'{k[1]}_{k[0]}': float(v) for k, v in e1m.items() if k[0] in (10, 100, 1000, 10000)})
    for meth in ['Residual', 'SysID']:
        g = R8[R8.method == meth].sort_values('N')
        a.errorbar(g.N * (1.06 if meth == 'SysID' else 0.94), g.success, yerr=[g.success - g.lo, g.hi - g.success],
                   fmt='o-', ms=3, lw=1.2, capsize=1.5, color=COL[meth], label=meth)
    a.axhline(orc, color=COL['Oracle'], ls=':', lw=1, label='True simulator')
    a.axhline(0.8 * orc, color='k', ls='--', lw=0.6, label='80% of true')
    a.axhline(drv, color=COL['DR'], ls='--', lw=1, label='DR')
    a.set_xscale('log'); a.set_ylim(-0.03, 1.03); a.set_xlabel('target transitions $N$')
    a.set_title('link mass +60%' if ax == 'm' else 'latency +4 steps')
axs8[0].set_ylabel('MPC success rate')
h, l = axs8[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=6, ncol=3, loc='upper center')
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.83)); fig.savefig(f'{FIG}/mpc.pdf'); plt.close(fig)

# ------------------------------------------------------------------ latency shift references
lat_ref = []
for reg in ['in', 'out']:
    for sz in ['small', 'large']:
        o = DR[(DR.regime == reg) & (DR.axis == 'lat') & (DR['size'] == sz)]
        lat_ref.append(dict(regime=reg, size=sz,
                            exact=float(o[o.method == 'Oracle'].nrmse10_shift.iloc[0]),
                            constrained=float(o[o.method == 'Oracle4_shift'].nrmse10.iloc[0])))
S['lat_shift_refs'] = lat_ref

# ------------------------------------------------------------------ E9 independent bases / test set
e9 = []
for tag in ['_b0', '_b1', '_b2']:
    f = f'{OUT}/E9{tag}.csv'
    if not os.path.exists(f):
        continue
    D = pd.read_csv(f)
    c = D[(D.mech == 'none') & (D.outl == 0) & (D.axis == 'm') & (D['size'] == 'small') & (D.N == 1000)]
    cl = c.set_index(['seed', 'method']).nrmse10
    out = dict(base=tag)
    for (ax, sz), g in D[(D.mech == 'none') & (D.outl == 0) & (D.imputer == 'none')].groupby(['axis', 'size']):
        if g.N.nunique() < 7:
            continue
        cc, lo, hi, nv = crossover(g)
        out[f'cross_{ax}_{sz}'] = (cc, lo, hi)
        g2 = g[g.N >= 1000]
        piv = g2.pivot_table(index=['N', 'seed'], columns='method', values='nrmse10')
        out[f'res_better_N1000plus_{ax}_{sz}'] = int((piv.groupby('N').mean().Residual < piv.groupby('N').mean().SysID).sum())
    for imp, g in D[D.mech == 'MCAR'].groupby('imputer'):
        for meth, h in g.groupby('method'):
            rr = h.set_index('seed').nrmse10 / cl.xs(meth, level='method')
            out[f'MCAR10_{imp}_{meth}'] = boot(rr)
    for meth, h in D[D.outl == 0.01].groupby('method'):
        out[f'spike1_{meth}'] = boot(h.set_index('seed').nrmse10 / cl.xs(meth, level='method'))
    out['clean_res'] = float(cl.xs('Residual', level='method').mean()); out['clean_sys'] = float(cl.xs('SysID', level='method').mean())
    e9.append(out)
S['E9'] = e9

json.dump(S, open(f'{OUT}/stats.json', 'w'), indent=1, default=float)
print('analysis done')
