"""Paper tables generated from result files. -> results/tables/*.md (the paper quotes them verbatim)
Usage: .venv/bin/python src/tables.py"""
import json

import numpy as np
import pandas as pd

from datasets import DATA, ROOT

OUT = ROOT / 'results/tables'
DS = {'mot17': 'MOT17 train', 'dancetrack': 'DanceTrack val'}
TRK = {'sort': 'SORT', 'bytetrack': 'ByteTrack', 'ocsort': 'OC-SORT', 'botsort': 'BoT-SORT'}


def f(x, d=2):
    return f'{100 * x:.{d}f}'


def rng(s):
    return f'{f(s.min())}–{f(s.max())}'


def oracle():
    rows = ['| Dataset | Boxes removed | AssA = PoseTrack21 FragA | Faithful FragA, one gap | '
            'Faithful FragA, scattered | CLEAR Frag, one gap | CLEAR Frag, scattered |', '|' + '---|' * 7]
    for d in DS:
        o = json.loads((ROOT / 'results/oracle' / f'{d}.json').read_text())
        for r in sorted({float(k.split('_')[0][1:]) for k in o}):
            b, s = o[f'r{r}_block'], o[f'r{r}_scatter']
            assert abs(b['AssA'] - s['AssA']) < 1e-12
            rows.append(f"| {DS[d]} | {100 * s['dropped'] / s['gt_boxes']:.2f}% | {f(s['AssA'])} | {f(b['FragA'])} | "
                        f"{f(s['FragA'])} | {b['CLEAR_Frag']} | {s['CLEAR_Frag']} |")
    return rows


DET = {'yolo26s': 'YOLO26-S', 'yolo26x': 'YOLO26-X'}


def pval(x, B=10000):
    """One-sided bootstrap p with the +1 correction; values at the floor 1/(B+1) are marked."""
    return f'{x:.4f}' + (' (floor)' if np.isclose(x, 1 / (B + 1)) else '')


def groups(csv, label):
    o = pd.read_csv(ROOT / csv)
    rows = [f'| Dataset | Detector | Outputs | HOTA | FA-HOTA faithful | FA-HOTA PoseTrack21 | '
            f'Gap (PoseTrack21 minus faithful) | Smallest 95% lower bound | Sequences with a positive gap |',
            '|' + '---|' * 9]
    for (d, det), g in o.groupby(['dataset', 'detector'], sort=False):
        pos = f'{g.n_seq_dH_pos.min()}/{g.n_seqs.iloc[0]} in every output' if (g.n_seq_dH_pos == g.n_seqs).all() \
            else f'{g.n_seq_dH_pos.min()} to {g.n_seq_dH_pos.max()} of {g.n_seqs.iloc[0]}'
        rows.append(f'| {DS[d]} | {DET.get(det, det)}{label} | {len(g)} | {rng(g.HOTA)} | {rng(g["FA-HOTA"])} | '
                    f'{rng(g["FA-HOTA_PT21"])} | {rng(g.dH)} | {f(g.dH_lo.min())} | {pos} |')
    return rows


def per_output():
    """Supplementary per-output table: confirmatory outputs, then the added YOLO26 outputs."""
    rows = ['| Set | Dataset | Detector | Tracker | Output | HOTA | DetA | AssA | FragA | FragA PT21 | FragA floor | '
            'FA-HOTA | FA-HOTA PT21 | Gap [95% CI] | One-sided p | Sequences with positive gap | IDF1 | IDSW | '
            'CLEAR Frag |', '|' + '---|' * 19]
    for csv, st in (('results/analysis/outputs.csv', 'confirmatory'), ('results/analysis_added/outputs.csv', 'added')):
        o = pd.read_csv(ROOT / csv)
        for _, r in o.iterrows():
            rows.append(f"| {st} | {DS[r.dataset]} | {DET.get(r.detector, r.detector)} | {TRK[r.tracker]} | {r.post} | "
                        f"{f(r.HOTA)} | {f(r.DetA)} | {f(r.AssA)} | {f(r.FragA)} | {f(r.FragA_PT21)} | "
                        f"{f(r.FragA_floor)} | {f(r['FA-HOTA'])} | {f(r['FA-HOTA_PT21'])} | {f(r.dH)} "
                        f"[{f(r.dH_lo)}, {f(r.dH_hi)}] | {pval(r.p_dH)} | {r.n_seq_dH_pos}/{r.n_seqs} | "
                        f"{f(r.IDF1)} | {int(r.IDSW)} | {int(r.Frag)} |")
    return rows


def tau():
    rows = ['| Set | Dataset | Detector | Kendall tau, faithful vs PoseTrack21 FA-HOTA [95% CI] | '
            'Kendall tau, faithful FA-HOTA vs HOTA [95% CI] |', '|' + '---|' * 5]
    for path, st in (('results/analysis/summary.json', 'confirmatory'), ('results/analysis_added/summary.json', 'added')):
        sm = json.loads((ROOT / path).read_text())
        for k in [k for k in sm if k.endswith('/tau_FAHOTA_vs_PT21')]:
            d, det = k.split('/')[:2]
            a, b = sm[k], sm[f'{d}/{det}/tau_FAHOTA_vs_HOTA']
            rows.append(f'| {st} | {DS[d]} | {DET.get(det, det)} | {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}] | '
                        f'{b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}] |')
    return rows


def correlations():
    sm = json.loads((ROOT / 'results/analysis/summary.json').read_text())
    rows = ['| Dataset | Detector | Kendall tau, FragA vs negated CLEAR Frag per track | '
            'Kendall tau, FragA vs negated IDSW per track | same, PoseTrack21 FragA | '
            'Kendall tau, FA-HOTA vs HOTA | same, PoseTrack21 FA-HOTA |', '|' + '---|' * 7]
    for k in [k for k in sm if k.endswith('/E2_kendall_FragA_vs_negFM')]:
        d, det = k.split('/')[:2]
        g = lambda n: f"{sm[f'{d}/{det}/{n}']:.2f}"  # noqa: E731
        rows.append(f'| {DS[d]} | {det} | {g("E2_kendall_FragA_vs_negFM")} | {g("E2_kendall_FragA_vs_negIDSW")} | '
                    f'{g("E2_kendall_FragA_PT21_vs_negIDSW")} | {g("kendall_FA-HOTA_vs_HOTA")} | '
                    f'{g("kendall_FA-HOTA_PT21_vs_HOTA")} |')
    rows += ['', '| Dataset | Spearman, FragA vs CLEAR Frag per track | same, PoseTrack21 | '
             'Spearman, FragA vs IDSW per track | same, PoseTrack21 | Spearman, FA-HOTA vs HOTA (pooled) | '
             'same, PoseTrack21 (pooled) |', '|' + '---|' * 7]
    for d in DS:
        g = lambda n: f"{sm[f'{d}/{n}']:.2f}"  # noqa: E731
        rows.append(f'| {DS[d]} | {g("E2_spearman_FragA_vs_FMpertrack")} | {g("E2_spearman_FragA_PT21_vs_FMpertrack")} | '
                    f'{g("E2_spearman_FragA_vs_IDSWpertrack")} | {g("E2_spearman_FragA_PT21_vs_IDSWpertrack")} | '
                    f'{sm[f"{d}/E2_spearman_FAHOTA_vs_HOTA"]:.4f} | {sm[f"{d}/E2_spearman_FAHOTA_PT21_vs_HOTA"]:.4f} |')
    return rows


def published():
    rows = ['| Benchmark | Tracker | Class | Sequences (videos) | HOTA | FA-HOTA faithful | FA-HOTA PoseTrack21 | '
            'Gap [95% CI over videos] |', '|' + '---|' * 8]
    for p in sorted((ROOT / 'results/published').glob('*.json')):
        j = json.loads(p.read_text())
        for c, x in j['classes'].items():
            if x['HOTA'] == 0:
                continue
            bench, trk = p.stem.replace('-', ' '), {'qdtrack': 'QDTrack'}.get(j['tracker'], j['tracker'])
            rows.append(f"| {bench} | {trk} | {c} | {x['n_seqs']} ({x['n_videos']}) | {f(x['HOTA'])} | "
                        f"{f(x['FA-HOTA'])} | {f(x['FA-HOTA_PT21'])} | {f(x['dH'])} "
                        f"[{f(x['dH_ci'][0])}, {f(x['dH_ci'][1])}] |")
    return rows


def dropout():
    d = pd.read_csv(ROOT / 'results/analysis/dropout.csv')
    rows = ['| Dataset | Tracker | Dropped | Faithful FragA (floor) | Fall in faithful FragA (seed SD) | '
            'Fall in PoseTrack21 FragA | Fall in HOTA | Difference [95% CI] | One-sided Holm p |', '|' + '---|' * 9]
    for _, r in d.iterrows():
        rows.append(f"| {DS[r.dataset]} | {TRK[r.tracker]} | {100 * r.rate:.0f}% | {f(r.FragA_r)} "
                    f"({f(r.FragA_floor_r)}) | {f(r.D_FragA)} ({f(r.seed_sd_FragA)}) | {f(r.D_FragA_PT21)} | "
                    f"{f(r.D_HOTA)} | {f(r.DiD)} [{f(r.DiD_lo)}, {f(r.DiD_hi)}] | {r.p_holm:.4f} |")
    return rows


def interp(csv):
    i = pd.read_csv(ROOT / csv)
    rows = ['| Dataset | Detector | Tracker | Boxes added | Change in faithful FragA | Change in PoseTrack21 FragA | '
            'Change in AssA | Difference [95% CI] | One-sided Holm p |', '|' + '---|' * 9]
    for _, r in i.iterrows():
        p = '' if np.isnan(r.get('p_holm', np.nan)) else f'{r.p_holm:.4f}'
        rows.append(f"| {DS[r.dataset]} | {DET.get(r.detector, r.detector)} | {TRK[r.tracker]} | "
                    f"{100 * r.boxes_added_frac:.2f}% | {f(r.dFragA)} | {f(r.dFragA_PT21)} | {f(r.dAssA)} | "
                    f"{f(r.DiD)} [{f(r.DiD_lo)}, {f(r.DiD_hi)}] | {p or 'not tested'} |")
    return rows


def mutants():
    m = json.loads((ROOT / 'results/mutation_check.json').read_text())
    names = {'pt21_rule': 'PoseTrack21 rule (break only on identity change)', 'fna_only': 'drop the predicted-identity '
             'break (step 4, second test)', 'fpa_only': 'drop the ground-truth break (step 4, first test)',
             'window_lower_inclusive': 'window includes the earlier match frame', 'window_skip_first':
             'window skips the first frame after the earlier match', 'alpha_strict': 'strict threshold cut (s > alpha)',
             'sort_time_first': 'sort by frame before identities', 'den_without_overlap':
             'denominator without the overlap term'}
    a, b = m['A_short'], m['B_long_ties']
    rows = [f"| Mutation | Short sequences, cases caught of {a['cases']} | Long sequences with ties, cases caught of "
            f"{b['cases']} |", '|' + '---|' * 3]
    for k, v in names.items():
        rows.append(f"| {v} | {a['mutants_caught'][k]} | {b['mutants_caught'][k]} |")
    return rows


def track_lengths():
    """Ground-truth track lengths after the evaluator's filter (MOT17 pedestrians with consider flag 1)."""
    import glob
    rows = ['| Dataset | Tracks | Boxes | Mean boxes per track | Median |', '|' + '---|' * 5]
    for d, pat, filt in (('mot17', 'MOT17/train/*-FRCNN/gt/gt.txt', True),
                         ('dancetrack', 'dancetrack/val/*/gt/gt.txt', False)):
        lens = []
        for fn in glob.glob(str(DATA / pat)):
            g = np.loadtxt(fn, delimiter=',', ndmin=2)
            g = g[(g[:, 6] == 1) & (g[:, 7] == 1)] if filt else g
            lens += list(np.unique(g[:, 1], return_counts=True)[1])
        rows.append(f'| {DS[d]} | {len(lens)} | {sum(lens)} | {np.mean(lens):.0f} | {np.median(lens):.0f} |')
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tabs = dict(track_lengths=track_lengths(), oracle=oracle(), groups=groups('results/analysis/outputs.csv', ''),
                groups_added=groups('results/analysis_added/outputs.csv', ' (added)'), published=published(),
                per_output=per_output(), tau=tau(), correlations=correlations(), mutants=mutants(),
                dropout=dropout(), interp=interp('results/analysis/interp.csv'),
                interp_added=interp('results/analysis_added/interp.csv'))
    for k, v in tabs.items():
        (OUT / f'{k}.md').write_text('\n'.join(v) + '\n')
        print(f'## {k}\n' + '\n'.join(v) + '\n')


if __name__ == '__main__':
    main()
