"""Mutation check: is the literal-reference test able to catch plausible bugs in the vectorised fragment code?

A copy of FAHOTA's per-threshold core is run with one deliberate bug at a time (the unmutated copy is first asserted
equal to FAHOTA itself), and each mutant is scored against the literal set-based reference. Two regimes:
  A  the 300 short random sequences of test_literal_reference (3-24 frames, <= 5 gt / 7 tracker ids);
  B  100 long random sequences (50-200 frames, <= 4 gt / 6 tracker ids) whose similarities sometimes equal a threshold
     exactly, so that the strict-cut mutant can show.
A case is one (sequence, threshold); a mutant is caught in a case when its FragA differs from the reference.
Usage: .venv/bin/python tests/mutation_check.py            compare with results/mutation_check.json (read-only)
       .venv/bin/python tests/mutation_check.py --out      (re)write results/mutation_check.json (scripts/reproduce.sh)
"""
import json
import sys

import numpy as np

import test_fahota as T
from fahota import EPS, FAHOTA, FAHOTACompare, fragment_sizes

ALPHAS = FAHOTA().array_labels
MUTANTS = ['pt21_rule', 'fna_only', 'fpa_only', 'window_lower_inclusive', 'window_skip_first', 'alpha_strict',
           'sort_time_first', 'den_without_overlap']
OUT = T.ROOT / 'results/mutation_check.json'


def core(data, alpha, mutant=None):
    """FAHOTA's per-threshold FragA (src/fahota.py eval_sequence loop), with an optional single mutation."""
    mt, gt_seen, tracker_seen, gt_id_count, tracker_id_count = FAHOTA._matches(data)
    if mt is None:
        return 0.0
    mt_t, mt_g, mt_p, mt_s = mt
    mask = mt_s > alpha if mutant == 'alpha_strict' else mt_s >= alpha - EPS
    if not mask.any():
        return 0.0
    keys = (mt_g[mask], mt_p[mask], mt_t[mask]) if mutant == 'sort_time_first' else (mt_t[mask], mt_p[mask], mt_g[mask])
    order = np.lexsort(keys)
    t, g, p = mt_t[mask][order], mt_g[mask][order], mt_p[mask][order]
    mc = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
    np.add.at(mc, (g, p), 1)
    den = gt_id_count[g, 0] + tracker_id_count[0, p] - (0 if mutant == 'den_without_overlap' else mc[g, p])
    if mutant == 'pt21_rule':
        o = np.lexsort((t, g))
        frag = np.empty(len(t))
        frag[o] = fragment_sizes(np.r_[True, (g[o][1:] != g[o][:-1]) | (p[o][1:] != p[o][:-1])])
        return float(np.sum(frag / den) / len(t))
    lo = {'window_lower_inclusive': t[:-1], 'window_skip_first': np.minimum(t[:-1] + 2, t[1:])}.get(mutant, t[:-1] + 1)
    same_pair = (g[1:] == g[:-1]) & (p[1:] == p[:-1])
    g_between = gt_seen[g[:-1], t[1:]] - gt_seen[g[:-1], lo] > 0
    p_between = tracker_seen[p[:-1], t[1:]] - tracker_seen[p[:-1], lo] > 0
    brk = ~same_pair | (False if mutant == 'fpa_only' else g_between) | (False if mutant == 'fna_only' else p_between)
    return float(np.sum(fragment_sizes(np.r_[True, brk]) / den) / len(t))


def literal_pairs(data, alpha):
    """Pair-level literal reference: for each (g, p) pair, label every frame in which g or p appears as a TPA or as
    an FNA/FPA frame, cut the TPA runs, and give each TP of a run of size s the value s / (|TPA|+|FNA|+|FPA|).
    Equivalent to T.literal_reference (asserted on regime A) but fast enough for long sequences."""
    mt = FAHOTA._matches(data)[0]
    frames = data['num_timesteps']
    m = [dict() for _ in range(frames)]
    if mt is not None:
        for t, g, p, s in zip(*mt):
            if s >= alpha - EPS:
                m[t][int(g)] = int(p)
    gt_at = [set(map(int, x)) for x in data['gt_ids']]
    trk_at = [set(map(int, x)) for x in data['tracker_ids']]
    pairs = {(g, p) for d in m for g, p in d.items()}
    total, n_tp = 0.0, sum(len(d) for d in m)
    if not n_tp:
        return 0.0
    for g, p in pairs:
        labels = [m[s].get(g) == p for s in range(frames) if g in gt_at[s] or p in trk_at[s]]
        n_g = sum(g in x for x in gt_at)
        n_p = sum(p in x for x in trk_at)
        tpa = sum(labels)
        den = n_g + n_p - tpa
        run = 0
        for lab in labels + [False]:
            if lab:
                run += 1
            elif run:
                total += run * run / den
                run = 0
    return total / n_tp


def long_seq(rng):
    data = T.random_seq(rng, T=int(rng.integers(50, 201)), n_gt=int(rng.integers(1, 5)), n_trk=int(rng.integers(1, 7)))
    for sim in data['similarity_scores']:  # some similarities sit exactly on a threshold
        tie = rng.random(sim.shape) < 0.2
        sim[tie] = rng.choice(ALPHAS, tie.sum())
    return data


def run():
    out = {}
    for regime, n, make in (('A_short', 300, lambda r: T.random_seq(r)), ('B_long_ties', 100, long_seq)):
        rng = np.random.default_rng(1 if regime == 'A_short' else 5)
        caught = dict.fromkeys(MUTANTS, 0)
        cases = seqs_with_tp = seqs_lower = 0
        for _ in range(n):
            data = make(rng)
            r = FAHOTACompare().eval_sequence(data)
            seqs_with_tp += bool((r['HOTA_TP'] > 0).any())
            seqs_lower += bool((r['FragA'] < r['FragA_PT21'] - 1e-12).any())
            for a, alpha in enumerate(ALPHAS):
                ref = literal_pairs(data, alpha)
                assert np.isclose(core(data, alpha), r['FragA'][a]), 'unmutated copy differs from FAHOTA'
                assert np.isclose(ref, r['FragA'][a]), 'pair-level reference differs from FAHOTA'
                if regime == 'A_short':
                    assert np.isclose(ref, T.literal_reference(data, alpha)['FragA']), 'references disagree'
                cases += 1
                for mu in MUTANTS:
                    caught[mu] += not np.isclose(core(data, alpha, mu), ref)
        out[regime] = dict(random_sequences=n, cases=cases, sequences_with_tp=seqs_with_tp,
                           sequences_where_faithful_below_pt21=seqs_lower, mutants_caught=caught)
    return out


if __name__ == '__main__':
    res = run()
    print(json.dumps(res, indent=1))
    if '--out' in sys.argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1))
    else:
        if not OUT.exists():
            sys.exit(f'{OUT.relative_to(T.ROOT)} is missing: it ships with the repository (or run scripts/reproduce.sh, '
                     'which writes it with --out); the counts above were computed but not compared')
        stored = json.loads(OUT.read_text())
        assert stored == res, 'results differ from results/mutation_check.json'
        print('matches results/mutation_check.json')
