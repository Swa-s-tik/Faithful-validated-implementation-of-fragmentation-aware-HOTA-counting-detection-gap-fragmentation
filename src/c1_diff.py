"""Compare a clean re-run of scripts/reproduce.sh (made in a separate copy of the repository) with this repository.

Compared: every file under results/ (tracks, per-sequence metrics, oracle, cross-check, published re-scoring,
dropout, analyses, tables, example.json, mutation_check.json, pt21_multiclass.json, and the added-output trees), every
figure under figures/. Files are compared byte for byte; JSON/CSV
that differ are compared again with wall-time fields removed. -> results/c1_diff.json
Usage: .venv/bin/python src/c1_diff.py <copy_dir>"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMING = ('t_hota', 't_fahota', 't_fahota_compare', 't_pt21', 'seconds')
TREES = ('results', 'figures')
SKIP = ('results/c1_diff.json',)  # this report itself


def numbers(x, key=''):
    if isinstance(x, dict):
        return {k2: v for k, v in x.items() if k not in TIMING for k2, v in numbers(v, f'{key}/{k}').items()}
    if isinstance(x, list):
        return {k2: v for i, v in enumerate(x) for k2, v in numbers(v, f'{key}[{i}]').items()}
    return {key: x}


def load(p):
    if p.suffix == '.json':
        return numbers(json.loads(p.read_text()))
    if p.suffix == '.csv':
        return {f'{i}/{k}': v for i, r in enumerate(csv.DictReader(open(p))) for k, v in r.items()}
    return None


def files(base):
    out = {str(p.relative_to(base)) for t in TREES for p in (base / t).rglob('*') if p.is_file()
           and not str(p.relative_to(base)).startswith('results/timing_runs/')}  # archived timings are inputs
    return out - set(SKIP)


def main(copy):
    a_files, b_files = files(ROOT), files(copy)
    rel_copy = copy.relative_to(ROOT) if copy.is_relative_to(ROOT) else copy.name  # no absolute paths in results
    out = dict(copy=str(rel_copy), compared=0, identical_bytes=0, equal_without_timings=[], differing=[],
               missing_in_rerun=sorted(a_files - b_files), extra_in_rerun=sorted(b_files - a_files),
               by_kind={})
    for rel in sorted(a_files & b_files):
        a, b = ROOT / rel, copy / rel
        kind = rel.split('/')[0] if rel.split('/')[0] != 'results' else '/'.join(rel.split('/')[:2])
        k = out['by_kind'].setdefault(kind, dict(compared=0, identical=0))
        out['compared'] += 1
        k['compared'] += 1
        if a.read_bytes() == b.read_bytes():
            out['identical_bytes'] += 1
            k['identical'] += 1
            continue
        da, db = load(a), load(b)
        if da is not None and da == db:
            out['equal_without_timings'].append(rel)
            continue
        if da is not None and db is not None and da.keys() == db.keys():
            diff = []
            for key in da:
                try:
                    d = abs(float(da[key]) - float(db[key]))
                except (TypeError, ValueError):
                    d = 0.0 if da[key] == db[key] else float('inf')
                if d > 0:
                    diff.append((key, d))
            out['differing'].append(dict(file=rel, n_values=len(da), n_diff=len(diff),
                                         max_abs=max(d for _, d in diff) if diff else 0.0, examples=diff[:3]))
        else:
            out['differing'].append(dict(file=rel, note='not comparable numerically (binary or structure differs)'))
    (ROOT / 'results/c1_diff.json').write_text(json.dumps(out, indent=1, default=str))
    print({k: (v if isinstance(v, (int, str)) else len(v)) for k, v in out.items() if k != 'by_kind'})
    for k, v in sorted(out['by_kind'].items()):
        print(f'  {k:<32} {v["identical"]}/{v["compared"]} identical')
    for d in out['differing'][:20]:
        print(d)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve())
