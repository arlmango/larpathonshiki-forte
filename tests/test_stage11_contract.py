"""Synthetic-only regression checks for group isolation and weak-class weighting."""
import copy
import hashlib
import json
import numpy as np
import pytest
from tools import stage11_fit as F


def units():
    rng = np.random.default_rng(17)
    result = []
    for y, n in [(1, 2), (1, 3), (0, 7), (0, 13)]:
        c = rng.normal(0, .05, (n, 4)); c[:, 0] += 2 * y - 1
        result.append({'y': y, 'C': c, 'X': c.copy(), 't': np.arange(n), 'ivs': []})
    return result


def test_heldout_features_and_labels_do_not_change_training():
    u = units(); a = F.train(u, [0, 2]); changed = copy.deepcopy(u)
    for i in [1, 3]:
        changed[i]['C'][:] = 1e9; changed[i]['X'][:] = -1e9
        changed[i]['y'] = 1 - changed[i]['y']; changed[i]['ivs'] = [(99, 100)]
    b = F.train(changed, [0, 2])
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert min(F.scores(a, u[0])) > max(F.scores(a, u[2]))


def test_unequal_row_counts_keep_classes_and_clips_balanced(monkeypatch):
    seen = {}
    def capture(X, y, w, lam):
        seen.update(y=y, w=w, lam=lam)
        return np.zeros(X.shape[1]), 0., True
    monkeypatch.setattr(F.H, 'fit_logreg', capture)
    F.train(units(), [0, 1, 2, 3])
    y, w = seen['y'], seen['w']
    assert np.isclose(w[y == 1].sum(), w[y == 0].sum())
    assert np.isclose(w[:2].sum(), w[2:5].sum())
    assert np.isclose(w[5:12].sum(), w[12:].sum())
    assert seen['lam'] == 10


def test_score_formula_preserves_candidate_order():
    u = units()[0]; m = F.train(units(), [0, 2])
    shuffled = {**u, 'C': u['C'][::-1]}
    np.testing.assert_allclose(F.scores(m, shuffled), F.scores(m, u)[::-1], rtol=0, atol=1e-12)


def test_diagnosis_rejects_changed_saved_input_before_loading_features(tmp_path, monkeypatch):
    from tools import stage11_diagnose as D
    base = tmp_path / 'research_models/stage11'
    base.mkdir(parents=True)
    source = tmp_path / 'synthetic_input.txt'
    source.write_text('original synthetic input')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (base / 'input_manifest.json').write_text(json.dumps({str(source): digest}))
    for name in ('weights.npz', 'config.json', 'summary.json'):
        (base / name).write_bytes(b'synthetic placeholder')
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()}
    (base / 'artifact_manifest.json').write_text(json.dumps(manifest))
    source.write_text('changed synthetic input')
    monkeypatch.setattr(F.A, 'lp', lambda: str(tmp_path))
    def unexpected_load():
        pytest.fail('features must not load after input integrity failure')
    monkeypatch.setattr(F, 'load_units', unexpected_load)
    with pytest.raises(F.Refused, match='saved input integrity failed'):
        D.diagnose()
