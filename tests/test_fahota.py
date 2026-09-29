"""Correctness checks for src/fahota.py. Run: .venv/bin/python -m pytest -q tests/  (or as a script).

1. HOTA Fig. 5 oracle (printed values and closed forms) and the detection-gap / spurious-association cases.
2. A literal per-TP reference built from the paper's set definitions (TPA, FNA, FPA, "no FNAs or FPAs between
   them") agrees with the vectorised FAHOTA on every alpha, on random sequences with random similarities.
3. FragA_PT21 / FA-HOTA_PT21 equal PoseTrack21's own hota.py (verbatim, src/third_party/posetrack21/hota.py) on every alpha.
4. `_matches` reproduces HOTA's own Hungarian calls exactly.
5. Properties: FragA_floor <= FragA <= FragA_PT21 <= AssA; FA-HOTA <= FA-HOTA_PT21frag <= FA-HOTA_PT21 <= HOTA; perfect tracker = 1;
   invariance to id relabelling and to time reversal.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fahota import EPS, FAHOTA, FAHOTACompare  # noqa: E402  (applies the numpy alias shim first)
import trackeval  # noqa: E402
from trackeval.metrics import hota as hota_mod  # noqa: E402


def load_pt21():
    """PoseTrack21's hota.py, verbatim, imported as a trackeval.metrics module so its relative imports resolve."""
    spec = importlib.util.spec_from_file_location('trackeval.metrics.pt21_hota', ROOT / 'src/third_party/posetrack21/hota.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HOTA


PT21 = load_pt21()
FIELDS = ('AssA', 'FragA', 'HOTA', 'FA-HOTA')


def seq(gt, trk, sim=None):
    """TrackEval preprocessed data from per-timestep id lists; similarity 1 everywhere unless given."""
    gt, trk = [np.array(x, int) for x in gt], [np.array(x, int) for x in trk]
    if sim is None:
        sim = [np.ones((len(g), len(p))) for g, p in zip(gt, trk)]
    return dict(num_timesteps=len(gt), gt_ids=gt, tracker_ids=trk,
                num_gt_ids=int(max([x.max() for x in gt if len(x)], default=-1)) + 1,
                num_tracker_ids=int(max([x.max() for x in trk if len(x)], default=-1)) + 1,
                num_gt_dets=sum(map(len, gt)), num_tracker_dets=sum(map(len, trk)), similarity_scores=sim)


def one_gt(pred):
    return seq([[0]] * len(pred), [[] if p is None else [p] for p in pred])


FIG5 = {'A': [0, 0, 0, 0, 1, 1, 1, 1], 'B': [0, 0, 1, 1, 0, 0, 1, 1], 'C': [0, 0, 1, 1, 2, 2, 3, 3]}
PRINTED = {'A': (.5, .5, .71, .71), 'B': (.5, .25, .71, .59), 'C': (.25, .25, .5, .5)}
EXACT = {'A': (.5, .5, .5 ** .5, .5 ** .5), 'B': (.5, .25, .5 ** .5, .125 ** .25), 'C': (.25, .25, .5, .5)}
GAP = one_gt([0, 0, 0, None, None, 0, 0, 0])
FPA = seq([[0], [0], [0], [1], [0], [0], [0], [0]], [[0]] * 8)


def mean(res, fields):
    return tuple(float(np.mean(res[f])) for f in fields)


def random_seq(rng, T=None, n_gt=None, n_trk=None):
    """Random sequence: ids present on random subsets of frames, random similarities (some zero)."""
    T = T or int(rng.integers(3, 25))
    n_gt, n_trk = n_gt or int(rng.integers(1, 6)), n_trk or int(rng.integers(1, 8))
    pg, pp = rng.uniform(0.3, 1.0, n_gt), rng.uniform(0.2, 1.0, n_trk)
    gt = [np.flatnonzero(rng.random(n_gt) < pg) for _ in range(T)]
    trk = [np.flatnonzero(rng.random(n_trk) < pp) for _ in range(T)]
    ug = np.unique(np.concatenate(gt)) if any(map(len, gt)) else np.zeros(0, int)
    ut = np.unique(np.concatenate(trk)) if any(map(len, trk)) else np.zeros(0, int)
    gt = [np.searchsorted(ug, x) for x in gt]  # contiguous ids, as TrackEval's preprocessing leaves them
    trk = [np.searchsorted(ut, x) for x in trk]
    sim = [np.where(rng.random((len(g), len(p))) < 0.3, 0.0, rng.random((len(g), len(p)))) for g, p in zip(gt, trk)]
    return seq(gt, trk, sim)


def literal_reference(data, alpha):
    """Eqs. 32-34 from the set definitions, one TP at a time. Matching is HOTA's (FAHOTA._matches, checked in
    test_matches_equal_hota); everything after it is independent of the vectorised code."""
    mt = FAHOTA._matches(data)[0]
    T = data['num_timesteps']
    m = [dict() for _ in range(T)]  # m[t][g] = p
    if mt is not None:
        for t, g, p, s in zip(*mt):
            if s >= alpha - EPS:
                m[t][int(g)] = int(p)
    inv = [{p: g for g, p in d.items()} for d in m]
    gt_at = [set(map(int, x)) for x in data['gt_ids']]
    trk_at = [set(map(int, x)) for x in data['tracker_ids']]
    tps = [(t, g, p) for t in range(T) for g, p in m[t].items()]
    n_tp = len(tps)
    n_fn, n_fp = data['num_gt_dets'] - n_tp, data['num_tracker_dets'] - n_tp
    if n_tp == 0:
        return dict(AssA=0.0, FragA=0.0, FAssA=0.0, DetA=0.0)
    A, F = [], []
    for t0, g, p in tps:
        label = {}  # frame -> 'TPA' or 'X' (an FNA and/or FPA of c)
        for s in range(T):
            if m[s].get(g) == p:
                label[s] = 'TPA'
            elif g in gt_at[s] or p in trk_at[s]:
                label[s] = 'X'  # g present but not matched to p (FNA) and/or p present but not matched to g (FPA)
        tpa = sum(v == 'TPA' for v in label.values())
        fna = sum(g in gt_at[s] and m[s].get(g) != p for s in range(T))
        fpa = sum(p in trk_at[s] and inv[s].get(p) != g for s in range(T))
        events = sorted(label)
        i = events.index(t0)
        lo = i
        while lo > 0 and label[events[lo - 1]] == 'TPA':
            lo -= 1
        hi = i
        while hi < len(events) - 1 and label[events[hi + 1]] == 'TPA':
            hi += 1
        den = tpa + fna + fpa
        A.append(tpa / den)
        F.append((hi - lo + 1) / den)
    A, F = np.array(A), np.array(F)
    return dict(AssA=A.mean(), FragA=F.mean(), FAssA=np.sqrt(A * F).mean(), DetA=n_tp / (n_tp + n_fn + n_fp))


def test_fig5():
    for name, pred in FIG5.items():
        got = mean(FAHOTA().eval_sequence(one_gt(pred)), FIELDS)
        assert tuple(round(v, 2) for v in got) == PRINTED[name], (name, got)
        assert np.allclose(got, EXACT[name]), (name, got)


def test_gap_and_fpa_cases():
    assert np.allclose(mean(FAHOTA().eval_sequence(GAP), ('AssA', 'FragA')), (6 / 8, 3 / 8))
    assert np.allclose(mean(FAHOTA().eval_sequence(FPA), ('FragA',)), ((3 * 3 + 4 * 4 + 1) / 64,))
    # PT21's own code: the gap and the spurious association are invisible (FragA = AssA on GAP)
    assert np.allclose(mean(PT21().eval_sequence(GAP), ('FragA', 'AssA')), (6 / 8, 6 / 8))
    assert np.allclose(mean(PT21().eval_sequence(FPA), ('FragA',)), ((7 * 7 + 1) / 64,))
    for name, pred in FIG5.items():  # every break in Fig. 5 is an id switch, so PT21 agrees there
        assert np.allclose(mean(PT21().eval_sequence(one_gt(pred)), ('FragA', 'FA-HOTA')), EXACT[name][1::2]), name


def test_literal_reference(n=300):
    rng = np.random.default_rng(1)
    for i in range(n):
        data = random_seq(rng)
        res = FAHOTA().eval_sequence(data)
        for a, alpha in enumerate(FAHOTA().array_labels):
            ref = literal_reference(data, alpha)
            for f in ('AssA', 'FragA', 'FAssA', 'DetA'):
                assert np.isclose(res[f][a], ref[f]), (i, alpha, f, res[f][a], ref[f])
            assert np.isclose(res['FA-HOTA'][a], np.sqrt(ref['DetA'] * ref['FAssA']))


def test_pt21_mirror_equals_pt21_code(n=300):
    rng = np.random.default_rng(2)
    for i in range(n):
        data = random_seq(rng)
        mine, theirs = FAHOTACompare().eval_sequence(data), PT21().eval_sequence(data)
        for f in ('HOTA', 'DetA', 'AssA'):
            assert np.allclose(mine[f], theirs[f]), (i, f)
        assert np.allclose(mine['FragA_PT21'], theirs['FragA']), (i, mine['FragA_PT21'], theirs['FragA'])
        assert np.allclose(mine['FA-HOTA_PT21'], theirs['FA-HOTA']), i


def test_matches_equal_hota(n=100):
    rng = np.random.default_rng(3)
    lsa = hota_mod.linear_sum_assignment
    for _ in range(n):
        data, calls = random_seq(rng), []
        hota_mod.linear_sum_assignment = lambda m: calls.append(lsa(m)) or calls[-1]
        try:
            trackeval.metrics.HOTA().eval_sequence(data)
        finally:
            hota_mod.linear_sum_assignment = lsa
        mt = FAHOTA._matches(data)[0]
        ref_g, ref_p, it = [], [], iter(calls)
        for g, p in zip(data['gt_ids'], data['tracker_ids']):
            if len(g) and len(p):
                r, c = next(it)
                ref_g.append(g[r])
                ref_p.append(p[c])
        if mt is None:
            assert not ref_g
            continue
        assert np.array_equal(mt[1], np.concatenate(ref_g)) and np.array_equal(mt[2], np.concatenate(ref_p))


def test_properties(n=300):
    rng = np.random.default_rng(4)
    for i in range(n):
        data = random_seq(rng)
        r = FAHOTACompare().eval_sequence(data)
        tol = 1e-12
        assert (r['FragA'] <= r['FragA_PT21'] + tol).all() and (r['FragA_PT21'] <= r['AssA'] + tol).all(), i
        assert (r['FragA'] <= r['FragA_FNAonly'] + tol).all() and (r['FragA'] <= r['FragA_FPAonly'] + tol).all(), i
        assert (r['FragA_floor'] <= r['FragA'] + tol).all(), i  # every fragment has at least one TP
        assert (r['FragA_FNAonly'] <= r['FragA_PT21'] + tol).all(), i  # PT21 breaks are a subset of FNA breaks
        assert (r['FA-HOTA'] <= r['FA-HOTA_PT21frag'] + tol).all(), i
        assert (r['FA-HOTA_PT21frag'] <= r['FA-HOTA_PT21'] + tol).all(), i
        assert (r['FA-HOTA_PT21'] <= r['HOTA'] + tol).all(), i
        # id relabelling (random permutations of gt and tracker ids)
        pg, pt = rng.permutation(data['num_gt_ids']), rng.permutation(data['num_tracker_ids'])
        perm = dict(data, gt_ids=[pg[x] for x in data['gt_ids']], tracker_ids=[pt[x] for x in data['tracker_ids']])
        # time reversal
        rev = dict(data, gt_ids=data['gt_ids'][::-1], tracker_ids=data['tracker_ids'][::-1],
                   similarity_scores=data['similarity_scores'][::-1])
        for other in (perm, rev):
            o = FAHOTACompare().eval_sequence(other)
            for f in ('HOTA', 'FragA', 'FA-HOTA', 'FragA_PT21', 'FA-HOTA_PT21'):
                assert np.allclose(r[f], o[f]), (i, f)
        # perfect tracker: tracker = gt, similarity 1
        perfect = seq(data['gt_ids'], data['gt_ids'])
        if perfect['num_gt_dets']:
            p = FAHOTACompare().eval_sequence(perfect)
            for f in ('HOTA', 'FragA', 'FA-HOTA', 'FragA_PT21', 'FA-HOTA_PT21'):
                assert np.allclose(p[f], 1.0), (i, f, p[f])


def test_combine_sequences():
    metric = FAHOTA()
    res = metric.combine_sequences({k: metric.eval_sequence(one_gt(p)) for k, p in FIG5.items()})
    assert np.allclose(res['FragA'], (.5 + .25 + .25) / 3)
    assert np.allclose(res['FA-HOTA'], np.sqrt(np.mean([.5, .125 ** .5, .25])))


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
