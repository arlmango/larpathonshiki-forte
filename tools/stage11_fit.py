"""Frozen stage11 exploratory logistic head. Local arrays only; stdout aggregates only."""
import io
import json
import os
import time
from pathlib import Path
import numpy as np
from tools import stage8_audit as A, stage9_head as H, stage10_weak as W
from tools.local_safe import Refused, run_guarded

ROOT = Path(__file__).resolve().parents[1]
DISCLOSURE = ('Research only; 2 positive clips and 12 provider weak negatives; repeated train-set reuse; '
              'no independent test; holdout closed; production fail-closed. No automatic next experiment.')


def require(ok, why):
    if not ok:
        raise Refused(why)


def save(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)


def pack(obj):
    return json.dumps(obj, sort_keys=True, indent=2, allow_nan=False).encode()


def load_units():
    base = Path(A.lp()) / 'train_probe' / 'stage11'
    require(A.fsha(base / 'manifest.json')[:16] == '785de8bcfd1c481c', 'unapproved feature manifest')
    man = json.loads((base / 'manifest.json').read_text())
    require(man['version'] == 2 and man['prefix_sec'] == 1, 'wrong feature version')
    require(man['code_sha16'] == A.fsha(ROOT / 'tools/stage11_points.py')[:16], 'feature code changed')
    sp, digest = A.snapshot_spec('C02', None, None)
    require(man['snapshot_sha16'] == digest[:16], 'snapshot mismatch')
    pos, neg, _ = A.train_sets('C02')
    lab, _ = H.supervision(A.read_snapshot(sp, digest), pos, neg)
    require(len(lab) == 2 and len(neg) == 12 and not (set(lab) & neg), 'unexpected supervision')
    expected = {s + '.feat.npz' for s in set(lab) | neg}
    require(set(man['files']) == expected, 'feature unit mismatch')
    U, inputs = [], {str(base / 'manifest.json'): A.fsha(base / 'manifest.json'), str(sp): digest}
    for name in sorted(expected):
        path = base / name
        require(A.fsha(path)[:16] == man['files'][name], 'feature hash mismatch')
        sha = name.split('.')[0]
        y = int(sha in lab)
        cache = Path(A.lp()) / 'train_probe/cache_emb_siglip2' / (sha + '.npz')
        inputs[str(path)] = A.fsha(path)
        inputs[str(cache)] = A.fsha(cache)
        c = H.load_emb(sha)
        require(c is not None, 'missing cache')
        P, dur = c
        with np.load(path, allow_pickle=False) as z:
            t, Z, mask = z['t_pre'], z['z_pre'], z['pre_in_label']
            require(int(z['y']) == y and len(z['t_lab_out']) == 0, 'label-dependent support')
            require(len(t) > 0 and np.all(np.diff(t) > 0) and np.all((t >= 0) & (t < 1)), 'bad prefix')
            require(np.isfinite(Z).all() and Z.shape == (len(t), 768), 'bad features')
            ivs = [(v['start'], v['end']) for v in lab[sha]['ivs']] if y else []
            require(all(b > a for a, b in ivs), 'zero width interval')
            require(np.array_equal(mask, H.in_any(t, ivs)), 'mask mismatch')
            require(mask.any() if y else not mask.any(), 'missing supervision')
            # Candidate support uses every prefix frame plus 20 fixed cache grid rows, for ALL clips.
            ix = W.grid(len(P['t']))
            cand = np.vstack([Z, P['z'][ix]])
            times = np.concatenate([t, P['t'][ix]])
            X = Z[mask] if y else cand
            U.append({'y': y, 'X': X, 'C': cand, 't': times, 'ivs': ivs, 'dur': dur})
    require(len(U) == 14, 'unexpected unit count')
    for rel in ['splits/C02.json', 'labels/C02.imported.jsonl', 'inventory_map.json']:
        path = Path(A.lp()) / rel
        inputs[str(path)] = A.fsha(path)
    require(sum(len(u['X']) for u in U if u['y']) == 5, 'unexpected positive coverage')
    return U, inputs


def train(U, indices):
    # Only training-unit arrays enter scaler, weights and fitting.
    selected = [U[i] for i in indices]
    require({u['y'] for u in selected} == {0, 1}, 'training class absent')
    med, sd = H.scaler([u['C'] for u in selected])
    X = np.vstack([u['X'] for u in selected])
    y = np.concatenate([np.full(len(u['X']), u['y']) for u in selected])
    w = np.concatenate([np.full(len(u['X']), 1 / len(u['X'])) for u in selected])
    w = np.where(y == 1, w / w[y == 1].sum(), w / w[y == 0].sum()) * len(y) / 2
    beta, b0, conv = H.fit_logreg((X - med) / sd, y, w, lam=10)
    require(bool(conv), 'fit did not converge')
    require(np.isfinite(beta).all() and np.isfinite(b0), 'nonfinite fit')
    return {'beta': beta, 'b0': np.asarray(b0), 'med': med, 'sd': sd, 'converged': np.asarray(conv)}


def scores(m, u):
    return ((u['C'] - m['med']) / m['sd']) @ m['beta'] + m['b0']


def evaluate(U, deadline):
    ps = [i for i, u in enumerate(U) if u['y']]
    ns = [i for i, u in enumerate(U) if not u['y']]
    margins = [[] for _ in U]
    wins, convergence, hits = [], [], []
    for p in ps:
        for n in ns:
            require(time.monotonic() < deadline, 'time budget exceeded')
            ix = [i for i in range(len(U)) if i not in (p, n)]
            m = train(U, ix)
            convergence.append(bool(m['converged']))
            threshold = max(float(scores(m, U[i]).max()) for i in ix if not U[i]['y'])
            sp, sn = float(scores(m, U[p]).max()), float(scores(m, U[n]).max())
            wins.append(float(sp > sn) + .5 * float(sp == sn))
            margins[p].append(sp - threshold)
            margins[n].append(sn - threshold)
            top = U[p]['t'][int(np.argmax(scores(m, U[p])))]
            hits.append(any(a <= top <= b for a, b in U[p]['ivs']))
    pred = [float(np.median(v)) > 0 for v in margins]
    tp = sum(pred[i] for i in ps); fp = sum(pred[i] for i in ns)
    fn, tn = len(ps) - tp, len(ns) - fp
    return {'group_folds': len(wins), 'leave_pair_out_auc': float(np.mean(wins)),
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'positive_clip_recall_ci95': W.clopper_pearson(tp, len(ps)),
            'negative_clip_specificity_ci95': W.clopper_pearson(tn, len(ns)),
            'exact_localisation_fold_hits': int(sum(hits)), 'localisation_folds_not_independent': True,
            'all_group_fits_converged': all(convergence), **W.duration_auc(U)}


def main(_argv=None):
    start = time.monotonic()
    out = Path(A.lp()) / 'research_models/stage11'
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Permanent attempt marker prevents accidental second experiment, including after interruption.
    save(out / 'attempt.json', pack({'started_unix': time.time(), 'disclosure': DISCLOSURE}))
    before = A.immutables('C02')
    require(before[0] == before[1], 'immutable baseline mismatch')
    U, inputs = load_units()
    for name in ['tools/stage11_fit.py', 'tools/stage11_points.py', 'tools/stage9_head.py',
                 'tools/stage10_weak.py', 'tools/stage8_audit.py', 'tools/local_safe.py',
                 'tools/annotate_intervals.py', 'tools/stage5_encoder.py', 'docs/PROTOCOL_STAGE11.md']:
        inputs[str(ROOT / name)] = A.fsha(ROOT / name)
    save(out / 'input_manifest.json', pack(inputs))
    result = evaluate(U, start + 300)
    require(time.monotonic() < start + 300, 'time budget exceeded')
    m = train(U, list(range(len(U))))
    require(A.immutables('C02') == before, 'immutable inputs changed')
    require(all(A.fsha(p) == h for p, h in inputs.items()), 'run inputs changed')
    buf = io.BytesIO(); np.savez(buf, **m)
    save(out / 'weights.npz', buf.getvalue())
    with np.load(out / 'weights.npz', allow_pickle=False) as z:
        restored = dict(z)
    require(all(np.array_equal(scores(m, u), scores(restored, u)) for u in U), 'reload mismatch')
    require(A.immutables('C02') == before, 'immutable inputs changed')
    require(all(A.fsha(p) == h for p, h in inputs.items()), 'run inputs changed')
    config = {'research_only': True, 'production_enabled': False, 'encoder': 'pinned existing SigLIP2 ONNX',
              'normalisation': 'L2 then own original cache median; training-only robust scaler',
              'candidates': 'every native PTS frame in [0,1) plus 20 rounded linspace original cache rows',
              'clip_score': 'max candidate logit', 'lambda': 10, 'positive_rows': 5,
              'positive_units': 2, 'weak_negative_units': 12, 'disclosure': DISCLOSURE,
              'decision_threshold': None, 'threshold_note': 'No deployment threshold selected'}
    save(out / 'config.json', pack(config))
    result.update(outcome='EXPLORATORY_MODEL_SAVED', research_only=True, holdout_opened=False,
                  final_fit_converged=bool(m['converged']), artifact_reload_exact=True,
                  weights_sha256=A.fsha(out / 'weights.npz'), runtime_s=round(time.monotonic() - start, 2),
                  disclosure=DISCLOSURE)
    save(out / 'summary.json', pack(result))
    manifest = {n: A.fsha(out / n) for n in ['weights.npz', 'config.json', 'summary.json', 'input_manifest.json']}
    save(out / 'artifact_manifest.json', pack(manifest))
    print('STAGE11 ' + json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(run_guarded(main))
