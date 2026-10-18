"""Read-only checks of the saved stage11 head. No fitting; aggregates only.

Run: python -B -m tools.stage11_diagnose
Requires the original local private artifacts. No holdout, video decoding or network.
Training diagnostics are resubstitution, never quality estimates.
"""
import json
from pathlib import Path
import numpy as np
from tools import stage11_fit as F
from tools.local_safe import run_guarded


def diagnose():
    base = Path(F.A.lp()) / 'research_models/stage11'
    manifest = json.loads((base / 'artifact_manifest.json').read_text())
    F.require(set(manifest) == {'weights.npz', 'config.json', 'summary.json', 'input_manifest.json'},
              'unexpected artifact manifest')
    F.require(all(F.A.fsha(base / n) == h for n, h in manifest.items()), 'artifact integrity failed')
    saved_inputs = json.loads((base / 'input_manifest.json').read_text())
    F.require(bool(saved_inputs) and all(F.A.fsha(p) == h for p, h in saved_inputs.items()),
              'saved input integrity failed')
    with np.load(base / 'weights.npz', allow_pickle=False) as z:
        m = dict(z)
    U, inputs = F.load_units()
    F.require(all(saved_inputs.get(p) == h for p, h in inputs.items()), 'input integrity failed')
    X = np.vstack([u['X'] for u in U])
    y = np.concatenate([np.full(len(u['X']), u['y']) for u in U])
    w = np.concatenate([np.full(len(u['X']), 1 / len(u['X'])) for u in U])
    w = np.where(y == 1, w / w[y == 1].sum(), w / w[y == 0].sum()) * len(y) / 2
    X = (X - m['med']) / m['sd']
    logits = X @ m['beta'] + m['b0']
    residual = w * (1 / (1 + np.exp(-logits)) - y)
    gradient = X.T @ residual + 10 * m['beta']
    return {'diagnostic_only_resubstitution_not_validation': True, 'no_refit': True,
            'artifact_integrity_ok': True, 'input_integrity_ok': True,
            'positive_training_rows': int((y == 1).sum()), 'negative_training_rows': int((y == 0).sum()),
            'positive_training_rows_correct_at_zero': int(((logits > 0) & (y == 1)).sum()),
            'negative_training_rows_correct_at_zero': int(((logits <= 0) & (y == 0)).sum()),
            'positive_total_weight': float(w[y == 1].sum()), 'negative_total_weight': float(w[y == 0].sum()),
            'gradient_beta_max_abs': float(abs(gradient).max()),
            'gradient_intercept_abs': float(abs(residual.sum())),
            'candidate_counts_range': [min(len(u['C']) for u in U), max(len(u['C']) for u in U)],
            'holdout_opened': False, 'production_enabled': False}


def main(_argv=None):
    print(json.dumps(diagnose(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(run_guarded(main))
