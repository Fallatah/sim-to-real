"""Run all experiments. Usage: python experiments.py <exp> ; results -> out/<exp>.csv"""
import sys, os, time, pickle, warnings, itertools
import numpy as np, pandas as pd
from joblib import Parallel, delayed
# Warnings are not suppressed globally. Residual-network convergence is recorded per fit
# (column res_converged); IterativeImputer's early-stopping warning is counted per run.
from sklearn.exceptions import ConvergenceWarning
import sim, models, faults, mpc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
SEEDS = range(10)
BUDGETS = [10, 30, 100, 300, 1000, 3000, 10000]
AXES = ['m', 'd', 'f', 'lat']
SIZES = {'m': {'small': 1.2, 'large': 1.6}, 'd': {'small': 1.2, 'large': 1.6},
         'f': {'small': 1.2, 'large': 1.6}, 'lat': {'small': 1, 'large': 4}}
SHIFT = {'m': {'small': 1.3, 'large': 1.9}, 'd': {'small': 1.3, 'large': 1.9},
         'f': {'small': 1.3, 'large': 1.9}, 'lat': {'small': 2, 'large': 6}}
IMPUTERS = ['listwise', 'median', 'median+ind', 'knn', 'regression', 'mice', 'interp']
# post-review additions (not preregistered): lag-aware imputer, and a clean control matched
# to the number of transitions that listwise deletion keeps
IMPUTERS_NEW = ['lagreg', 'clean_matched']
BASE_FILE = os.environ.get('BASE_FILE', 'base.pkl')     # independent bases for the multi-base check
TEST_SEED = int(os.environ.get('TEST_SEED', '12345'))   # held-out trajectory set

_BASE = None
def base():
    global _BASE
    if _BASE is None:
        _BASE = pickle.load(open(os.path.join(OUT, BASE_FILE), 'rb'))
    return _BASE

_TEST = {}
def testset(axis, value, unmodeled):
    key = (axis, value, unmodeled)
    if key not in _TEST:
        tp = dict(sim.NOMINAL); tp[axis] = value
        rng = np.random.default_rng(TEST_SEED)
        p = sim.params(40, **tp, unmodeled=unmodeled)
        Q, DQ, TAU = sim.rollout_logs(40, 80, p, rng)
        X, Y, _, _ = sim.transitions(Q, DQ, TAU)
        sd = np.concatenate([Q, DQ], 2).reshape(-1, 4).std(0)
        _TEST[key] = (Q, DQ, TAU, X, Y, sd)
    return _TEST[key]


def evaluate(model, axis, value, unmodeled):
    Q, DQ, TAU, X, Y, sd = testset(axis, value, unmodeled)
    e10, _ = models.multistep_nrmse(model, Q, DQ, TAU, state_sd=sd)
    r1, m1, r2 = models.onestep_metrics(model, X, Y, base().sc.ysd)
    return dict(nrmse10=e10, rmse1=r1, mae1=m1, r2=r2)


import zlib
def data_seed(*keys):
    return zlib.crc32(repr(keys).encode())


def run(exp, regime, axis, size, N, seed, mech='none', rate=0.0, imputer='none',
        outl=0.0, treat='raw', noise=0.0, shift=False, methods=('Residual', 'SysID')):
    """One cell x seed: collect target data, inject faults, clean, fit, evaluate."""
    unm = regime == 'out'
    value = SIZES[axis][size]
    tp = dict(sim.NOMINAL); tp[axis] = value
    rng = np.random.default_rng(data_seed(regime, axis, size, N, seed))
    R, TAU = faults.collect(N, tp, unm, rng)
    sd = R.reshape(-1, 4).std(0)
    frng = np.random.default_rng(data_seed('faults', mech, rate, outl, noise, seed))
    if noise > 0:
        R = faults.add_noise(R, noise, frng, sd)
    R_clean = R.copy()
    if outl > 0:
        R, _ = faults.add_spikes(R, outl, frng, sd)
    # Treatment is applied whenever requested, including on clean data (outl == 0), so the
    # clean-data rows measure what the treatment itself costs. (Fix: it used to sit inside
    # `if outl > 0`, which made the clean treatment controls identical to untreated runs.)
    if treat == 'winsorize':
        R = faults.winsorize(R)
    elif treat == 'discard':
        R = faults.mad_discard(R)
    if rate > 0:
        R = faults.apply_mask(R, faults.missing_mask(R, TAU, mech, rate, frng))
    n_imp_warn = 0
    if imputer == 'clean_matched':
        # clean transitions, randomly subsampled to the count listwise deletion keeps
        keep, _ = faults.to_sets([R], TAU, N)
        csets, _ = faults.to_sets([R_clean], TAU, N)
        Xc, Yc = csets[0]
        sel = np.random.default_rng(data_seed('match', mech, rate, seed)).choice(len(Xc), len(keep[0][0]), replace=False)
        sets, extra = [(Xc[sel], Yc[sel])], None
    else:
        imp = imputer if imputer != 'none' else 'listwise'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', ConvergenceWarning)
            Rs, miss = faults.impute(R, TAU, imp, seed)
        n_imp_warn = sum(issubclass(x.category, ConvergenceWarning) for x in w)
        sets, extra = faults.to_sets(Rs, TAU, N, miss, indicator=(imp == 'median+ind'))
    n_used = len(sets[0][0])
    rows = []
    base_info = dict(exp=exp, regime=regime, axis=axis, size=size, value=value, N=N, seed=seed,
                     mech=mech, rate=rate, imputer=imputer, outl=outl, treat=treat, noise=noise,
                     n_used=n_used, imp_warnings=n_imp_warn)
    for meth in methods:
        t0 = time.time()
        if n_used < 2:
            res = dict(nrmse10=np.nan, rmse1=np.nan, mae1=np.nan, r2=np.nan); param = np.nan; conv = np.nan
        else:
            conv = np.nan
            if meth == 'Residual':
                m = models.ResidualModel(base(), sets, seed=seed, extra=extra); param = np.nan
                conv = float(np.mean(m.converged))
            else:
                Xs = np.concatenate([s[0] for s in sets]); Ys = np.concatenate([s[1] for s in sets])
                m = models.SysIDModel(axis, Xs, Ys, base().sc.ysd); param = m.p.get(axis, np.nan)
            res = evaluate(m, axis, value, unm)
            if shift:
                res['nrmse10_shift'] = evaluate(m, axis, SHIFT[axis][size], unm)['nrmse10']
        rows.append({**base_info, 'method': meth, 'param': param, 'res_converged': conv,
                     'fit_s': time.time() - t0, **res})
    return rows


def dr_rows():
    rows = []
    for regime, axis, size in itertools.product(['in', 'out'], AXES, ['small', 'large']):
        unm = regime == 'out'
        res = evaluate(models.DRModel(base()), axis, SIZES[axis][size], unm)
        res['nrmse10_shift'] = evaluate(models.DRModel(base()), axis, SHIFT[axis][size], unm)['nrmse10']
        tp = dict(sim.NOMINAL); tp[axis] = SIZES[axis][size]
        orc = evaluate(models.OracleModel(tp, unm), axis, SIZES[axis][size], unm)
        # shifted target: exact reference (true history from full logs) and the best
        # predictor that only sees the NLAG-step torque history in the features
        tps = dict(sim.NOMINAL); tps[axis] = SHIFT[axis][size]
        Qs, DQs, TAUs, _, _, sds = testset(axis, SHIFT[axis][size], unm)
        orc['nrmse10_shift'] = models.exact_multistep_nrmse(tps, unm, Qs, DQs, TAUs, state_sd=sds)
        orc4 = evaluate(models.OracleModel(tps, unm, constrained=True), axis, SHIFT[axis][size], unm)
        rows.append(dict(regime=regime, axis=axis, size=size, method='DR', **res))
        rows.append(dict(regime=regime, axis=axis, size=size, method='Oracle', **orc))
        rows.append(dict(regime=regime, axis=axis, size=size, method='Oracle4_shift', **orc4))
    return rows


def jobs(exp):
    J = []
    if exp == 'E1':   # budget sweep, all mismatch types/sizes, both regimes
        for regime, axis, size, N, s in itertools.product(['in', 'out'], AXES, ['small', 'large'], BUDGETS, SEEDS):
            J.append(dict(exp=exp, regime=regime, axis=axis, size=size, N=N, seed=s, shift=(N == 1000)))
    if exp == 'E2':   # missingness x imputer at reference (out-of-class); in-class at 30%
        for mech, rate, imp, s in itertools.product(['MCAR', 'MAR', 'MNAR'], [0.1, 0.3, 0.5], IMPUTERS, SEEDS):
            J.append(dict(exp=exp, regime='out', axis='m', size='small', N=1000, seed=s, mech=mech, rate=rate, imputer=imp))
        for mech, imp, s in itertools.product(['MCAR', 'MAR', 'MNAR'], IMPUTERS, SEEDS):
            J.append(dict(exp=exp, regime='in', axis='m', size='small', N=1000, seed=s, mech=mech, rate=0.3, imputer=imp))
    if exp == 'E3':   # outliers x treatment, and Gaussian noise
        for o, tr, s in itertools.product([0.01, 0.05, 0.10], ['raw', 'winsorize', 'discard'], SEEDS):
            J.append(dict(exp=exp, regime='out', axis='m', size='small', N=1000, seed=s, outl=o, treat=tr))
        for tr, s in itertools.product(['winsorize', 'discard'], SEEDS):   # treatment cost on clean data
            J.append(dict(exp=exp, regime='out', axis='m', size='small', N=1000, seed=s, outl=0.0, treat=tr))
        for nz, s in itertools.product([0.01, 0.05], SEEDS):
            J.append(dict(exp=exp, regime='out', axis='m', size='small', N=1000, seed=s, noise=nz))
    if exp == 'E6':   # budget x mechanism cross (MICE fixed in advance)
        for mech, N, s in itertools.product(['MCAR', 'MAR', 'MNAR'], BUDGETS, SEEDS):
            J.append(dict(exp=exp, regime='out', axis='m', size='small', N=N, seed=s, mech=mech, rate=0.3, imputer='mice'))
    if exp == 'E7':   # robustness by mismatch type (small size)
        for axis, s in itertools.product(AXES, SEEDS):
            J.append(dict(exp=exp, regime='out', axis=axis, size='small', N=1000, seed=s, mech='MAR', rate=0.3, imputer='mice'))
            J.append(dict(exp=exp, regime='out', axis=axis, size='small', N=1000, seed=s, outl=0.05, treat='raw'))
    if exp == 'E2b':  # post-review: lag-aware imputer and matched-N clean control
        for mech, rate, imp, s in itertools.product(['MCAR', 'MAR', 'MNAR'], [0.1, 0.3, 0.5], IMPUTERS_NEW, SEEDS):
            J.append(dict(exp=exp, regime='out', axis='m', size='small', N=1000, seed=s, mech=mech, rate=rate, imputer=imp))
    if exp == 'E9':   # post-review: key findings re-run with an independent base and test set
        for axis, size, N, s in itertools.product(['m', 'f'], ['large', 'small'], BUDGETS, SEEDS):
            if (axis, size) in (('m', 'large'), ('f', 'small')):
                J.append(dict(exp=exp, regime='out', axis=axis, size=size, N=N, seed=s))
        for imp, s in itertools.product(['listwise', 'median', 'mice', 'lagreg'], SEEDS):
            J.append(dict(exp=exp, regime='out', axis='m', size='small', N=1000, seed=s, mech='MCAR', rate=0.1, imputer=imp))
        for s in SEEDS:
            J.append(dict(exp=exp, regime='out', axis='m', size='small', N=1000, seed=s))
            J.append(dict(exp=exp, regime='out', axis='m', size='small', N=1000, seed=s, outl=0.01, treat='raw'))
    return J


MPC_COND = {'m': 1.6, 'lat': 4}


def mpc_job(N, seed, method, axis='m'):
    tp = dict(sim.NOMINAL); tp[axis] = MPC_COND[axis]
    if method == 'Oracle':
        m = models.OracleModel(tp, True)
    elif method == 'DR':
        m = models.DRModel(base())
    else:
        rng = np.random.default_rng(data_seed('out', axis, 'large', N, seed))
        R, TAU = faults.collect(N, tp, True, rng)
        sets, _ = faults.to_sets([R], TAU, N)
        X, Y = sets[0]
        m = (models.ResidualModel(base(), sets, seed=seed) if method == 'Residual'
             else models.SysIDModel(axis, X, Y, base().sc.ysd))
    sr = mpc.success_rate(m, tp, True, 20, 777 + seed)
    return dict(exp='E8', axis=axis, N=N, seed=seed, method=method, success=sr)


if __name__ == '__main__':
    exp = sys.argv[1]
    t0 = time.time()
    if exp == 'base':
        # optional: python experiments.py base <k> trains an independent base (seed k) for the multi-base check
        bseed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        rng = np.random.default_rng(bseed)
        B = 4000
        p = sim.params(B, m=rng.uniform(0.7, 1.3, B), d=rng.uniform(0.7, 1.3, B),
                       f=rng.uniform(0.7, 1.3, B), lat=rng.integers(0, 2, B))
        Q, DQ, TAU = sim.rollout_logs(B, 50 + sim.NLAG, p, rng)
        X, Y, _, _ = sim.transitions(Q, DQ, TAU)
        b = models.BaseEnsemble(X, Y, max_iter=60, seed=bseed * 1000)
        name = 'base.pkl' if bseed == 0 else f'base{bseed}.pkl'
        pickle.dump(b, open(os.path.join(OUT, name), 'wb'))
        if bseed == 0:
            np.save(os.path.join(OUT, 'source_sample.npy'), np.concatenate([X[:20000], Y[:20000]], 1))
        print('base trained on', len(X), 'loss', [round(m.loss_, 4) for m in b.members])
    elif exp == 'DR':
        pd.DataFrame(dr_rows()).to_csv(os.path.join(OUT, 'DR.csv'), index=False)
    elif exp == 'E8':
        # mass +60% (preregistered condition) and latency +4 (post-review second condition)
        J = [(N, s, m, ax) for ax in ['m', 'lat'] for N in [10, 100, 1000, 10000] for s in range(5)
             for m in ['Residual', 'SysID']]
        J += [(0, s, m, ax) for ax in ['m', 'lat'] for s in range(5) for m in ['DR', 'Oracle']]
        rows = Parallel(n_jobs=2)(delayed(mpc_job)(*j) for j in J)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'E8.csv'), index=False)
    else:
        # Resumable: jobs run in NCHUNK chunks saved to out/parts/; finished chunks are skipped,
        # then the chunks are merged into out/<exp><tag>.csv.
        J = jobs(exp)
        tag = os.environ.get('OUT_TAG', '')
        nchunk = int(os.environ.get('NCHUNK', '20'))
        pdir = os.path.join(OUT, 'parts'); os.makedirs(pdir, exist_ok=True)
        files = []
        for c in range(nchunk):
            f = os.path.join(pdir, f'{exp}{tag}_{c:03d}.csv'); files.append(f)
            if os.path.exists(f):
                continue
            rows = Parallel(n_jobs=int(os.environ.get('NJOBS', '2')), batch_size=4)(delayed(run)(**j) for j in J[c::nchunk])
            pd.DataFrame([r for rr in rows for r in rr]).to_csv(f + '.tmp', index=False)
            os.replace(f + '.tmp', f)
            print(exp, tag, 'chunk', c, 'of', nchunk, round(time.time() - t0), 's', flush=True)
        pd.concat([pd.read_csv(f) for f in files]).sort_values(['method', 'seed'], kind='stable').to_csv(
            os.path.join(OUT, f'{exp}{tag}.csv'), index=False)
    print(exp, 'done in', round(time.time() - t0), 's')
