"""Pre-registered analysis: sequence bootstrap, H1-H3 and exploratory E1-E2, from results/metrics/<dataset>/*.json.

Every metric is recombined from per-sequence per-alpha quantities exactly as TrackEval combines sequences (sums of
TP/FN/FP and counts; TP-weighted averages of AssA-type fields, divided by max(1, TP)); the full-sample recombination is
asserted equal to TrackEval's own COMBINED_SEQ. Writes results/analysis/{summary.json, outputs.csv, pairs.csv,
interp.csv} and prints the headline tables.
Usage: .venv/bin/python src/analyze.py
"""
import csv
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from datasets import ADDED, ROOT, SUFFIX  # noqa: E402
B, SEED, CHUNK = int(os.environ.get('FAHOTA_BOOT_B', 10000)), 20260928, 250  # pre-registered B = 10000
TPW = ['AssA', 'FragA', 'FAssA', 'FragA_PT21', 'FAssA_PT21frag', 'FragA_FNAonly', 'FragA_FPAonly', 'FragA_floor']
SUMS = ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'Frag', 'MT', 'PT', 'ML', 'IDTP', 'IDFN', 'IDFP', 'GT_Dets', 'Dets']
REPORT = ['HOTA', 'DetA', 'AssA', 'FragA', 'FragA_PT21', 'FragA_FNAonly', 'FragA_FPAonly', 'FragA_floor', 'FA-HOTA',
          'FA-HOTA_PT21', 'FA-HOTA_PT21frag', 'IDF1', 'MOTA', 'IDSW', 'Frag', 'FM_per_track', 'IDSW_per_track']
TRACKERS = ['sort', 'bytetrack', 'ocsort', 'botsort']
METRICS = f'results/metrics{SUFFIX}'  # SUFFIX '_added': outputs added after pre-registration, analysed apart


def load(dataset):
    files = sorted((ROOT / METRICS / dataset).glob('*.json'))
    runs = {f.stem: json.loads(f.read_text()) for f in files}
    names = sorted(runs)
    seqs = sorted(runs[names[0]]['per_seq'])
    A = {k: np.array([[runs[n]['per_seq'][s][k] for s in seqs] for n in names])  # (outputs, seqs, alpha)
         for k in ['HOTA_TP', 'HOTA_FN', 'HOTA_FP'] + TPW}
    S = {k: np.array([[runs[n]['per_seq'][s][k] for s in seqs] for n in names]) for k in SUMS}  # (outputs, seqs)
    return names, seqs, A, S, runs


def combine(A, S, W):
    """W: (b, seqs) multiplicities -> {metric: (b, outputs)}, alpha-averaged like TrackEval's summary."""
    tp = np.einsum('bs,osa->boa', W, A['HOTA_TP'])
    fn, fp = (np.einsum('bs,osa->boa', W, A[k]) for k in ('HOTA_FN', 'HOTA_FP'))
    m = {k: np.einsum('bs,osa->boa', W, A[k] * A['HOTA_TP']) / np.maximum(1.0, tp) for k in TPW}
    det = tp / np.maximum(1.0, tp + fn + fp)
    out = {'DetA': det, 'HOTA': np.sqrt(det * m['AssA']), 'FA-HOTA': np.sqrt(det * m['FAssA']),
           'FA-HOTA_PT21': np.sqrt(det * np.sqrt(m['AssA'] * m['FragA_PT21'])),  # PT21 L211
           'FA-HOTA_PT21frag': np.sqrt(det * m['FAssA_PT21frag'])}
    out.update(m)
    out = {k: v.mean(-1) for k, v in out.items()}
    s = {k: W @ S[k].T for k in SUMS}  # (b, outputs)
    out['IDF1'] = s['IDTP'] / np.maximum(1.0, s['IDTP'] + 0.5 * s['IDFP'] + 0.5 * s['IDFN'])
    out['MOTA'] = (s['CLR_TP'] - s['CLR_FP'] - s['IDSW']) / np.maximum(1.0, s['CLR_TP'] + s['CLR_FN'])
    out['IDSW'], out['Frag'] = s['IDSW'], s['Frag']
    out['FM_per_track'] = s['Frag'] / np.maximum(1.0, s['MT'] + s['PT'] + s['ML'])
    out['IDSW_per_track'] = s['IDSW'] / np.maximum(1.0, s['MT'] + s['PT'] + s['ML'])
    return out


def bootstrap(A, S, n_seq, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seq, size=(B, n_seq))
    W = np.zeros((B, n_seq))
    np.add.at(W, (np.arange(B)[:, None], idx), 1)
    parts = [combine(A, S, W[i:i + CHUNK]) for i in range(0, B, CHUNK)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def p_one(d):  # H: d > 0; p = P(d <= 0), with the +1 correction
    return (1 + np.sum(d <= 0, axis=0)) / (d.shape[0] + 1)


def p_two(d):
    return np.minimum(1.0, 2 * np.minimum(p_one(d), (1 + np.sum(d >= 0, axis=0)) / (d.shape[0] + 1)))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o])
    out = np.empty_like(p)
    out[o] = np.minimum(1.0, adj)
    return out


def ci(x):
    return np.percentile(x, [2.5, 97.5], axis=0)


def parse(name):
    det, trk, post = name.split('__')
    return det, trk, post


def main():
    out_dir = ROOT / ('results/analysis' if B == 10000 else f'results/analysis_B{B}')  # B != 10000 = post hoc
    if ADDED:
        out_dir = ROOT / 'results/analysis_added'
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, pair_rows, interp_rows, summary = [], [], [], {}
    h1, pairs_f = [], {'PT21': [], 'HOTA': []}
    for di, dataset in enumerate(['mot17', 'dancetrack']):
        if not list((ROOT / METRICS / dataset).glob('*.json')):
            continue
        names, seqs, A, S, runs = load(dataset)
        full = combine(A, S, np.ones((1, len(seqs))))
        for k in ('HOTA', 'FragA', 'FragA_PT21', 'FA-HOTA', 'FA-HOTA_PT21', 'FA-HOTA_PT21frag'):  # == TrackEval
            te = np.array([np.mean(runs[n]['combined'][k]) for n in names])
            assert np.allclose(full[k][0], te, atol=1e-12), (dataset, k)
        bs = bootstrap(A, S, len(seqs), SEED + di)
        per = combine(A, S, np.eye(len(seqs)))  # each sequence on its own: sign consistency of the gaps
        # per-output table
        dH, dF = bs['FA-HOTA_PT21'] - bs['FA-HOTA'], bs['FragA_PT21'] - bs['FragA']
        for i, n in enumerate(names):
            det, trk, post = parse(n)
            r = dict(dataset=dataset, detector=det, tracker=trk, post=post, output=n, n_seqs=len(seqs))
            for k in REPORT:
                r[k] = full[k][0, i]
                r[k + '_lo'], r[k + '_hi'] = ci(bs[k][:, i])
            r['dH'], (r['dH_lo'], r['dH_hi']) = full['FA-HOTA_PT21'][0, i] - full['FA-HOTA'][0, i], ci(dH[:, i])
            r['dF'], (r['dF_lo'], r['dF_hi']) = full['FragA_PT21'][0, i] - full['FragA'][0, i], ci(dF[:, i])
            r['L211_part'] = full['FA-HOTA_PT21'][0, i] - full['FA-HOTA_PT21frag'][0, i]
            r['rule_part'] = full['FA-HOTA_PT21frag'][0, i] - full['FA-HOTA'][0, i]
            r['HOTA_minus_FAHOTA_PT21'] = full['HOTA'][0, i] - full['FA-HOTA_PT21'][0, i]
            r['HOTA_minus_FAHOTA'] = full['HOTA'][0, i] - full['FA-HOTA'][0, i]
            r['p_dH'] = p_one(dH[:, i])
            r['p_dH_at_floor'] = bool(np.isclose(r['p_dH'], 1 / (B + 1)))  # resolution floor of B resamples
            r['n_seq_dH_pos'] = int(np.sum(per['FA-HOTA_PT21'][:, i] - per['FA-HOTA'][:, i] > 0))
            r['n_seq_dF_pos'] = int(np.sum(per['FragA_PT21'][:, i] - per['FragA'][:, i] > 0))
            h1.append(r)
            rows.append(r)
        # pairs within groups
        dets = sorted({parse(n)[0] for n in names})
        for det in dets:
            g = [i for i, n in enumerate(names) if parse(n)[0] == det]
            for comp, key in (('PT21', 'FA-HOTA_PT21'), ('HOTA', 'HOTA')):
                tau = [kendalltau(bs['FA-HOTA'][b, g], bs[key][b, g])[0] for b in range(B)]  # all resamples
                t0 = kendalltau(full['FA-HOTA'][0, g], full[key][0, g])[0]
                summary[f'{dataset}/{det}/tau_FAHOTA_vs_{comp}'] = [t0, *np.percentile(tau, [2.5, 97.5])]
                for i, j in combinations(g, 2):
                    df, dc = bs['FA-HOTA'][:, i] - bs['FA-HOTA'][:, j], bs[key][:, i] - bs[key][:, j]
                    f0 = full['FA-HOTA'][0, i] - full['FA-HOTA'][0, j]
                    c0 = full[key][0, i] - full[key][0, j]
                    pairs_f[comp].append(dict(dataset=dataset, detector=det, a=names[i], b=names[j], comparator=comp,
                                              d_fahota=f0, d_fahota_lo=ci(df)[0], d_fahota_hi=ci(df)[1],
                                              d_comp=c0, d_comp_lo=ci(dc)[0], d_comp_hi=ci(dc)[1],
                                              p_fahota=p_two(df), p_comp=p_two(dc), opposite=bool(f0 * c0 < 0),
                                              frac_boot_opposite=float(np.mean(df * dc < 0))))
        # H3: interpolation, per (detector, tracker)
        for det in dets:
            for trk in TRACKERS:
                try:
                    i, j = names.index(f'{det}__{trk}__interp'), names.index(f'{det}__{trk}__raw')
                except ValueError:
                    continue
                did = (bs['FragA'][:, i] - bs['FragA'][:, j]) - (bs['FragA_PT21'][:, i] - bs['FragA_PT21'][:, j])
                added = full_boxes(dataset, f'{det}__{trk}__interp') / full_boxes(dataset, f'{det}__{trk}__raw') - 1
                interp_rows.append(dict(dataset=dataset, detector=det, tracker=trk, boxes_added_frac=added,
                                        dFragA=full['FragA'][0, i] - full['FragA'][0, j],
                                        dFragA_PT21=full['FragA_PT21'][0, i] - full['FragA_PT21'][0, j],
                                        dHOTA=full['HOTA'][0, i] - full['HOTA'][0, j],
                                        dAssA=full['AssA'][0, i] - full['AssA'][0, j],
                                        dFAHOTA=full['FA-HOTA'][0, i] - full['FA-HOTA'][0, j],
                                        dFAHOTA_PT21=full['FA-HOTA_PT21'][0, i] - full['FA-HOTA_PT21'][0, j],
                                        DiD=((full['FragA'][0, i] - full['FragA'][0, j]) -
                                             (full['FragA_PT21'][0, i] - full['FragA_PT21'][0, j])),
                                        DiD_lo=ci(did)[0], DiD_hi=ci(did)[1], p=p_one(did)))
        # E2: FragA vs FM per GT track, across this dataset's outputs
        fm = full['FM_per_track'][0]
        summary[f'{dataset}/E2_spearman_FragA_vs_FMpertrack'] = spearmanr(full['FragA'][0], fm)[0]
        summary[f'{dataset}/E2_spearman_FragA_PT21_vs_FMpertrack'] = spearmanr(full['FragA_PT21'][0], fm)[0]
        idsw = full['IDSW_per_track'][0]
        summary[f'{dataset}/E2_spearman_FragA_vs_IDSWpertrack'] = spearmanr(full['FragA'][0], idsw)[0]
        summary[f'{dataset}/E2_spearman_FragA_PT21_vs_IDSWpertrack'] = spearmanr(full['FragA_PT21'][0], idsw)[0]
        summary[f'{dataset}/E2_spearman_FAHOTA_vs_HOTA'] = spearmanr(full['FA-HOTA'][0], full['HOTA'][0])[0]
        summary[f'{dataset}/E2_spearman_FAHOTA_PT21_vs_HOTA'] = spearmanr(full['FA-HOTA_PT21'][0], full['HOTA'][0])[0]
        summary[f'{dataset}/E1_FragA_FNAonly_equals_FragA_all_outputs'] = bool(
            np.allclose(full['FragA_FNAonly'], full['FragA'], atol=1e-12))
        for det in dets:
            g = [i for i, n in enumerate(names) if parse(n)[0] == det]
            summary[f'{dataset}/{det}/E2_kendall_FragA_vs_negFM'] = kendalltau(full['FragA'][0, g], -fm[g])[0]
            for k, v in (('FragA', 'FragA'), ('FragA_PT21', 'FragA_PT21')):
                summary[f'{dataset}/{det}/E2_kendall_{k}_vs_negIDSW'] = kendalltau(full[v][0, g], -idsw[g])[0]
            for k in ('FA-HOTA', 'FA-HOTA_PT21'):  # within-group agreement with HOTA (pooling inflates it)
                summary[f'{dataset}/{det}/kendall_{k}_vs_HOTA'] = kendalltau(full[k][0, g], full['HOTA'][0, g])[0]
            hr = full['HOTA'][0, g]
            summary[f'{dataset}/{det}/HOTA_range'] = float(hr.max() - hr.min())

    # H1 decision
    p_adj = holm([r['p_dH'] for r in h1])
    for r, p in zip(h1, p_adj):
        r['p_dH_holm'] = p
    summary['H1_holds'] = bool(all(p < 0.05 for p in p_adj))
    summary['H1_n_outputs'] = len(h1)
    # H2 decision: Holm per comparator over all pairs x 2 metrics
    for comp, lst in pairs_f.items():
        adj = holm([d['p_fahota'] for d in lst] + [d['p_comp'] for d in lst])
        n = len(lst)
        for k, d in enumerate(lst):
            d['p_fahota_holm'], d['p_comp_holm'] = adj[k], adj[n + k]
            d['sig_reversal'] = bool(d['opposite'] and adj[k] < 0.05 and adj[n + k] < 0.05)
        summary[f'H2_{comp}_n_pairs'] = n
        summary[f'H2_{comp}_n_opposite'] = sum(d['opposite'] for d in lst)
        summary[f'H2_{comp}_n_sig_reversals'] = sum(d['sig_reversal'] for d in lst)
        pair_rows += lst
    summary['H2_holds'] = summary['H2_PT21_n_sig_reversals'] >= 1
    # H3 decision
    tested = [r for r in interp_rows if r['boxes_added_frac'] >= 0.005]
    adj = holm([r['p'] for r in tested]) if tested else []
    for r, p in zip(tested, adj):
        r['p_holm'] = p
    summary['H3_n_tested'] = len(tested)
    summary['H3_n_untested'] = len(interp_rows) - len(tested)
    summary['H3_holds'] = bool(tested and all(p < 0.05 for p in adj))

    for fname, lst in (('outputs.csv', rows), ('pairs.csv', pair_rows), ('interp.csv', interp_rows)):
        with open(out_dir / fname, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(lst[0]))
            w.writeheader()
            w.writerows([{k: (float(v) if isinstance(v, (np.floating, np.ndarray)) else v) for k, v in r.items()}
                         for r in lst])
    (out_dir / 'summary.json').write_text(json.dumps({k: (np.asarray(v).tolist()) for k, v in summary.items()},
                                                     indent=1))
    print(json.dumps({k: np.round(np.asarray(v, float), 4).tolist() if not isinstance(v, bool) else v
                      for k, v in summary.items()}, indent=0))


def full_boxes(dataset, output):
    return sum(sum(1 for _ in open(f)) for f in (ROOT / f'results/tracks{SUFFIX}' / dataset / output).glob('*.txt'))


if __name__ == '__main__':
    main()
