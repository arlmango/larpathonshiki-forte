"""Stage12: train-only temporal multiple-instance head; local inputs, aggregate stdout.

All 17 vendor-labelled clips train bags; two exact intervals add positive-only
auxiliary supervision. Inference candidates never depend on interval labels.
"""
import io
import json
import os
import time
from pathlib import Path
import numpy as np
from tools import stage8_audit as A, stage9_head as H
from tools.local_safe import Refused, run_guarded, select_split_units

ROOT = Path(__file__).resolve().parents[1]
DIM, L2, AUX, SMOOTH = 12, 0.1, 0.25, 0.01


def require(ok, reason):
    if not ok:
        raise Refused(reason)


def save(path, content):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, 'wb') as f:
        f.write(content)


def pack(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def load_units():
    from tools.import_labels import locate_xlsx
    units, counts = select_split_units(A.lp(), 'C02', 'train', dry_run=False)
    require(len(units) == 17 and not any(counts.values()), 'train source coverage failed')
    pos = {s for s, _, st, _ in units if st == 'reviewed_violation'}
    neg = {s for s, _, st, _ in units if st == 'reviewed_no_violation'}
    require(len(pos) == 5 and len(neg) == 12 and not pos & neg, 'train classes differ')
    sp, digest = A.snapshot_spec('C02', None, None)
    labels, _ = H.supervision(A.read_snapshot(sp, digest), pos, neg)
    require(len(labels) == 2, 'exact interval coverage differs')
    base = Path(A.lp())
    prefix_manifest = base / 'train_probe/stage11/manifest.json'
    require(A.fsha(prefix_manifest)[:16] == '785de8bcfd1c481c', 'prefix manifest differs')
    prefix_files = json.loads(prefix_manifest.read_text())['files']
    inputs = {str(sp): digest, str(prefix_manifest): A.fsha(prefix_manifest)}
    original, n = locate_xlsx(A.lp(), 'C02')
    require(n == 1 and Path(original).is_file(), 'original table unavailable')
    inputs[str(original)] = A.fsha(original)
    for rel in ['labels/C02.imported.jsonl', 'splits/C02.json', 'inventory_map.json']:
        inputs[str(base / rel)] = A.fsha(base / rel)
    records = [json.loads(line) for line in (base / 'labels/C02.imported.jsonl').read_text().splitlines()]
    provenance = {r.get('sha16') or r.get('sha') for r in records
                  if r.get('kind') == 'imported_clip_status' and r.get('source') == 'xlsx'}
    require(pos | neg <= provenance, 'clip labels lack original table provenance')
    U, fallback = [], 0
    for sha, source, status, endpoints in units:
        require(Path(source).resolve().is_relative_to(Path.home() / 'Downloads/open'), 'source outside approved root')
        inputs[str(source)] = A.fsha(source)
        cache = base / 'train_probe/cache_emb_siglip2' / (sha + '.npz')
        inputs[str(cache)] = A.fsha(cache)
        with np.load(cache, allow_pickle=False) as z:
            t = z['t'].astype(float); E = z['E_zone'].astype(float)
            dur = float(z['dur_dec']); fallback += int(z['zone_fallback'])
        require(E.shape == (len(t), 768) and len(t) >= 3 and np.isfinite(E).all()
                and np.isfinite(t).all() and np.all(np.diff(t) > 0) and dur > 0, 'invalid cache')
        P = H.point_prep(t, E)
        ivs = [(v['start'], v['end']) for v in labels.get(sha, {}).get('ivs', [])]
        az, at, aw = np.empty((0, 768)), np.empty(0), np.empty(0)
        if sha in labels:
            path = base / 'train_probe/stage11' / (sha + '.feat.npz')
            require(A.fsha(path)[:16] == prefix_files[path.name], 'exact feature hash differs')
            inputs[str(path)] = A.fsha(path)
            with np.load(path, allow_pickle=False) as z:
                mask = H.in_any(z['t_pre'], ivs)
                require(np.array_equal(mask, z['pre_in_label']), 'exact frame mask differs')
                az, at = z['z_pre'][mask].astype(float), z['t_pre'][mask].astype(float)
            aw = np.array([max(v['w'] for v in labels[sha]['ivs'] if v['start'] <= tt <= v['end']) for tt in at])
            require(len(at) > 0 and np.isfinite(az).all(), 'missing exact auxiliary frames')
        U.append({'y': int(sha in pos), 'z': P['z'], 't': t, 'dur': dur,
                  'aux_z': az, 'aux_t': at, 'aux_w': aw, 'ivs': ivs, 'endpoints': endpoints})
    cov = {'train_clips': len(U), 'vendor_positive': len(pos), 'vendor_negative': len(neg),
           'cache_coverage': len(U), 'finite_dims': 768, 'cached_rows_total': sum(len(u['t']) for u in U),
           'exact_interval_clips': len(labels), 'unknown_interval_positive_retained': len(pos)-len(labels),
           'exact_aux_positive_frames': sum(len(u['aux_t']) for u in U),
           'exact_interval_clips_with_candidate_hit': sum(bool(ivs and H.in_any(u['t'], ivs).any())
                                                        for u in U for ivs in [u['ivs']]),
           'zone_fallback': fallback, 'train_source_hashes_match': True,
           'sources_under_requested_downloads_open': True, 'original_xlsx_present': True,
           'unique_train_statuses_from_xlsx': len((pos | neg) & provenance), 'holdout_features_opened': False}
    return U, inputs, cov


def temporal(u, projection, auxiliary=False):
    """Offline clip centring, point + causal 2s mean + absolute deviation; no labels."""
    r = u['z'] @ projection
    q = u['aux_z'] @ projection if auxiliary else r
    ts = u['aux_t'] if auxiliary else u['t']
    context = []
    for i, tt in enumerate(ts):
        ix = (u['t'] <= tt) & (u['t'] > tt - 2.)
        context.append(r[ix].mean(0) if ix.any() else q[i])
    c = np.asarray(context).reshape(q.shape)
    return np.concatenate([q, c, np.abs(q-c)], axis=1)


def representation(U, indices):
    selected = [U[i] for i in indices]
    # Equal mass per training clip; no held-out arrays enter PCA or normalisation.
    X = np.vstack([u['z'] / np.sqrt(len(u['z']) * len(selected)) for u in selected])
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    projection = vt[:min(DIM, len(vt))].T
    fs = [temporal(u, projection) for u in selected]
    mean = np.mean([f.mean(0) for f in fs], axis=0)
    var = np.mean([((f-mean)**2).mean(0) for f in fs], axis=0)
    sd = np.maximum(np.sqrt(var), 1e-4)
    return {'projection': projection, 'mean': mean, 'sd': sd}


def features(u, model, auxiliary=False):
    return (temporal(u, model['projection'], auxiliary)-model['mean']) / model['sd']


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


def pooling(logits):
    # Temperature = 1. Normalised logsumexp, invariant to uniform row replication.
    maximum = np.max(logits)
    a = np.exp(logits-maximum)
    return float(maximum + np.log(a.mean())), a/a.sum()


def balanced_weights(y):
    y = np.asarray(y)
    require(set(y) == {0, 1}, 'training class absent')
    return np.where(y == 1, .5 / (y == 1).sum(), .5 / (y == 0).sum())


def objective(theta, bags, ys, auxiliaries, aux_weights):
    beta, bias = theta[:-1], theta[-1]
    loss = .5 * L2 * float(beta @ beta)
    gradient = np.r_[L2*beta, 0.]
    weights = balanced_weights(ys)
    positives = [i for i, y in enumerate(ys) if y and len(auxiliaries[i])]
    negatives = [i for i, y in enumerate(ys) if not y]
    for i, (X, y, weight) in enumerate(zip(bags, ys, weights)):
        logits = X @ beta + bias
        score, attention = pooling(logits)
        loss += weight * (np.logaddexp(0., score) - y*score)
        g = weight * (sigmoid(score)-y) * attention
        gradient += np.r_[X.T @ g, g.sum()]
        if len(X) > 1:
            delta = np.diff(X, axis=0)
            changes = delta @ beta
            loss += SMOOTH * float(np.mean(changes**2)) / len(bags)
            gradient[:-1] += 2*SMOOTH * (delta.T @ changes) / (len(changes)*len(bags))
        # No auxiliary negative targets for any frame of a positive clip.
        if i in positives:
            Z = auxiliaries[i]; f = Z @ beta + bias
            w = aux_weights[i]/aux_weights[i].sum()
            coefficient = AUX * .5 / len(positives)
            loss += coefficient * float(w @ np.logaddexp(0., -f))
            g = coefficient*w*(sigmoid(f)-1.)
            gradient += np.r_[Z.T @ g, g.sum()]
        elif not y and positives:
            coefficient = AUX * .5 / len(negatives)
            loss += coefficient * float(np.logaddexp(0., logits).mean())
            g = coefficient*sigmoid(logits)/len(logits)
            gradient += np.r_[X.T @ g, g.sum()]
    return float(loss), gradient


def minimise(fun, size, deadline):
    """One deterministic BFGS start with Armijo backtracking, no hyperparameter search."""
    theta = np.zeros(size); inverse = np.eye(size)
    value, grad = fun(theta)
    for iteration in range(500):
        require(time.monotonic() < deadline, 'fit deadline reached')
        require(np.isfinite(value) and np.isfinite(grad).all(), 'nonfinite optimisation')
        if np.max(np.abs(grad)) < 1e-6:
            return theta, {'converged': True, 'iterations': iteration, 'gradient_max_abs': float(abs(grad).max())}
        direction = -inverse @ grad
        if direction @ grad >= 0:
            direction = -grad; inverse = np.eye(size)
        rate = 1.
        for _ in range(35):
            candidate = theta + rate*direction
            new_value, new_grad = fun(candidate)
            if np.isfinite(new_value) and new_value <= value + 1e-4*rate*(grad@direction):
                break
            rate *= .5
        else:
            raise Refused('optimiser line search failed')
        step, change = candidate-theta, new_grad-grad
        curvature = float(step @ change)
        if curvature > 1e-12:
            rho = 1./curvature; V = np.eye(size)-rho*np.outer(step, change)
            inverse = V @ inverse @ V.T + rho*np.outer(step, step)
        else:
            inverse = np.eye(size)
        theta, value, grad = candidate, new_value, new_grad
    raise Refused('optimiser did not converge')


def train(U, indices, deadline):
    model = representation(U, indices)
    selected = [U[i] for i in indices]
    bags = [features(u, model) for u in selected]
    aux = [features(u, model, True) for u in selected]
    ys = [u['y'] for u in selected]
    fn = lambda theta: objective(theta, bags, ys, aux, [u['aux_w'] for u in selected])
    theta, info = minimise(fn, bags[0].shape[1]+1, deadline)
    model.update(beta=theta[:-1], bias=np.asarray(theta[-1]))
    return model, info


def inference(u, model):
    logits = features(u, model) @ model['beta'] + model['bias']
    score, _ = pooling(logits)
    return score, float(u['t'][int(np.argmax(logits))]), logits


def metrics(y, prediction, scores=None):
    y = np.asarray(y, bool); p = np.asarray(prediction, bool)
    tp, fp, fn, tn = [int(a.sum()) for a in [y&p, ~y&p, y&~p, ~y&~p]]
    result = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': tp/max(1,tp+fp),
              'recall': tp/max(1,tp+fn), 'f1': 2*tp/max(1,2*tp+fp+fn),
              'balanced_accuracy': .5*(tp/max(1,tp+fn)+tn/max(1,tn+fp))}
    if scores is not None:
        s = np.asarray(scores)
        result['auc'] = float(np.mean([(a>b)+.5*(a==b) for a in s[y] for b in s[~y]]))
    return result


def duration_baseline(U):
    from tools.stage7_core import fit_thr_mid
    predictions = []
    for k, u in enumerate(U):
        train_u = [v for i,v in enumerate(U) if i != k]
        y = [v['y'] for v in train_u]; d = np.array([v['dur'] for v in train_u])
        choices = []
        for sign in [1., -1.]:
            threshold = fit_thr_mid(sign*d, y)
            choices.append((metrics(y, sign*d >= threshold)['f1'], sign, threshold))
        _, sign, threshold = max(choices, key=lambda z:z[0])
        predictions.append(sign*u['dur'] >= threshold)
    result = metrics([u['y'] for u in U], predictions)
    result['method'] = 'LOO sign and F1 threshold fitted on training clips only'
    return result


def evaluate(U, deadline):
    scores, peaks, infos, folds = [], [], [], []
    for k in range(len(U)):
        indices = [i for i in range(len(U)) if i != k]
        model, info = train(U, indices, deadline)
        score, peak, _ = inference(U[k], model)
        scores.append(score); peaks.append(peak); infos.append(info); folds.append(model)
    y = [u['y'] for u in U]
    pos = [i for i,u in enumerate(U) if u['y']]
    exact = [i for i,u in enumerate(U) if u['ivs']]
    hit = lambda i,t: any(abs(t-v)<=5. for v in U[i]['endpoints'])
    logdur = np.log([u['dur'] for u in U])
    correlation = float(np.corrcoef(scores, logdur)[0,1]) if np.std(scores)>0 and np.std(logdur)>0 else None
    result = {'clip_oof': metrics(y, np.array(scores)>=0., scores), 'folds': len(U),
              'all_folds_converged': all(i['converged'] for i in infos),
              'max_gradient': max(i['gradient_max_abs'] for i in infos),
              'endpoint_hits_within_5s': sum(hit(i,peaks[i]) for i in pos), 'endpoint_positive_clips': len(pos),
              'exact_interval_hits': sum(any(a<=peaks[i]<=b for a,b in U[i]['ivs']) for i in exact),
              'exact_interval_clips': len(exact),
              'exact_candidate_support_hit_rate': float(np.mean([H.in_any(U[i]['t'],U[i]['ivs']).mean() for i in exact])) if exact else None,
              'always_positive': metrics(y, np.ones(len(U))), 'always_negative': metrics(y, np.zeros(len(U))),
              'duration_only': duration_baseline(U),
              'random_candidate_endpoint_hit_rate': float(np.mean([np.mean([hit(i,t) for t in U[i]['t']]) for i in pos])),
              'first_candidate_endpoint_hit_rate': float(np.mean([hit(i,U[i]['t'][0]) for i in pos])),
              'duration_auc_shorter_positive': metrics(y,np.ones(len(U)),[-u['dur'] for u in U])['auc'],
              'oof_score_logduration_pearson': correlation,
              'outcome': 'NO_GO_EXPLORATORY_ONLY', 'production_enabled': False,
              'holdout_opened': False, 'independent_validation': False,
              'go_gates_not_relaxed': True, 'permutation_and_independent_validation_not_performed': True}
    return result, folds


def main(argv=None):
    import sys
    args = sys.argv[1:] if argv is None else argv
    out = Path(A.lp()) / 'research_models/stage12'
    if args == ['--coverage']:
        _, _, cov = load_units(); print('COVERAGE '+json.dumps(cov,sort_keys=True)); return 0
    require(args == ['--run'], 'use --coverage or --run')
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    save(out/'attempt.json', pack({'started_unix':time.time(), 'single_attempt':True}))
    start = time.monotonic(); deadline = start+600
    immutable_before = A.immutables('C02')
    require(immutable_before[0] == immutable_before[1], 'immutable baseline differs')
    U, inputs, cov = load_units()
    for rel in ['tools/stage12_mil.py','tests/test_stage12_contract.py','docs/PROTOCOL_STAGE12.md',
                'tools/stage8_audit.py','tools/stage9_head.py','tools/local_safe.py','tools/stage7_core.py',
                'tools/import_labels.py','tools/annotate_intervals.py','tools/annotate.py','tools/evaluate.py']:
        inputs[str(ROOT/rel)] = A.fsha(ROOT/rel)
    save(out/'input_manifest.json',pack(inputs)); save(out/'coverage.json',pack(cov))
    print('COVERAGE '+json.dumps(cov,sort_keys=True),flush=True)
    result, folds = evaluate(U, deadline)
    model, final_info = train(U,list(range(len(U))),deadline)
    buf = io.BytesIO(); np.savez(buf,**model); save(out/'weights.npz',buf.getvalue())
    buf = io.BytesIO(); np.savez(buf,**{f'fold_{i}_{k}':v for i,m in enumerate(folds) for k,v in m.items()})
    save(out/'cv_weights.npz',buf.getvalue())
    with np.load(out/'weights.npz',allow_pickle=False) as z:
        restored = dict(z)
    reload_ok = all(np.array_equal(inference(u,model)[2],inference(u,restored)[2]) for u in U)
    require(reload_ok, 'reload differs')
    result.update(final_fit=final_info,artifact_reload_exact=reload_ok,training_clips=len(U),
                  final_fit_resubstitution=metrics([u['y'] for u in U],[inference(u,model)[0]>=0 for u in U]),
                  runtime_s=round(time.monotonic()-start,3),weights_sha256=A.fsha(out/'weights.npz'))
    config = {'method':'clip-balanced temporal logmeanexp MIL with exact-positive auxiliary loss',
              'encoder':'existing frozen pinned SigLIP2 zone cache', 'trainable_component':'temporal MIL head only',
              'pca_dims':DIM,'l2':L2,'auxiliary_weight':AUX,'temporal_smoothness':SMOOTH,'temperature':1.,
              'projection':'clip-balanced uncentered SVD of clip-median residuals',
              'normalization':'training clips equally weighted mean/std, std floor 1e-4',
              'optimizer':'zero-init BFGS Armijo, max 500 iterations, gradient infinity norm <1e-6',
              'temporal_context_seconds':2.,'research_logit_threshold':0.,'deployment_threshold':None,
              'production_enabled':False,'offline_full_clip_centring':True,
              'candidate_support':'original 1fps cache for every clip, independent of labels',
              'unknown_interval_positive_policy':'positive bag loss, no frame targets',
              'validation':'17 outer leave-one-clip-out fits; exploratory repeated train reuse',
              'disclosure':'vendor labels weak; no event type or norm inferred; holdout closed this stage'}
    require(all(A.fsha(p)==h for p,h in inputs.items()), 'inputs changed during fitting')
    require(A.immutables('C02') == immutable_before, 'immutables changed')
    save(out/'config.json',pack(config));save(out/'summary.json',pack(result))
    save(out/'artifact_manifest.json',pack({p.name:A.fsha(p) for p in out.iterdir() if p.is_file()}))
    print('STAGE12 '+json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(run_guarded(main))
