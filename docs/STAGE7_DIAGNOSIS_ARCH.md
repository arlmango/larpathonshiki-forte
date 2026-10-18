# STAGE 7 diagnosis — detection-architect view (independent of STAGE7_DIAGNOSIS_REVIEW.md)

Author: detection-architect (claude-opus-5-5), 2026-10-18 ~02:10 UTC. No code changes. No real-data run.
Inputs read: COORDINATION.md (stages 4–7), PROTOCOL_STAGE4/5, REVIEW_CRITICAL (S4-1..S4-8, C5-1..C5-10),
tools/stage4_probe.py, stage5_model.py, stage5_probe.py, stage5_encoder.py, train_probe.py, app/pipeline.py.
Only the aggregates in stage4_summary.json / stage5_summary.json were read: no per-clip rows, shas, paths,
frames, labels or embeddings. All experiments below are **synthetic** (numpy fixtures in the session scratch
directory `diag7a/`, with LOCAL_PRIVATE_DIR there). The parallel reviewer's document was not read.

## 1. Diagnosis as testable claims

Real aggregates used throughout:
- n = 17 clips, 5 positives.
- Duration min / median / max = 17.3 / 23.8 / 142.5 s.
- Duration AUC vs label 0.283, so positives are shorter. Duration-only nested F1 0.60, with "shorter" chosen in
  17/17 folds.
- Random-time hit 0.387 (0.424 over the scan support).
- Stage 5:
  - outer TP 3 / FP 12 / FN 2 / TN 0;
  - permutation p95: F1 0.667, hit 0.80; p50: F1 0.375, hit 0.40;
  - 9 distinct configs across folds; median |ρ(score, duration)| 0.33, max 0.59.
- Stage 4:
  - the config `objects_near` was chosen in 14/17 folds;
  - permutation p95: F1 0.727, hit 0.60.

| # | Claim | Evidence so far | Decisive test (local program, aggregates only) |
|---|---|---|---|
| D1 | **Scene identity dominates the stage-5 learning problem (C5-1).** Positive targets come from 5 clips and negatives are *every* step of 12 other clips. PCA-8 is fitted on all steps, so between-clip variance fills the components. The classifier then separates 5 scenes from 12 scenes, which cannot transfer under clip-LOO. Stage 4's `objects_near` (static COCO clutter near the zone) is the same failure in count form. | Real: 9 configs across folds; F1 0.30 < null median 0.375. **Synthetic (this pilot):** on a scene-only fixture (positives differ only by a static whole-clip shift, with no event), the *real* `stage5_model.nested` reaches outer F1 **0.91**, P(F1 > 0.667) = 1.00 (6 reps). On planted within-clip cues of amplitude 0.08 / 0.15 / 0.25 it gives only F1 0.48 / 0.43 / 0.48. It is sensitive to scene and nearly blind to the within-clip event. | η²(clip id) of each PCA-8 component, each prompt cosine and the raw E_zone/E_full dims, on the 17 caches (one number per family). **Prediction: η² > 0.8.** If η² < 0.5, D1 is refuted as the main cause. |
| D2 | **The target window assumes a visible ≤ 10 s event ending at the label.** That is unsupported. T1 4 end / 1 middle: one label is not an endpoint, so its [−10, +2] target is mis-centred. V1 4/5 unseen: the cue may be absent, off-zone, or an *omission* (the norm candidates C-20/C-31-4 are "recount in presence of the client", so a violation may be the *lack* of an action). A positive-appearance detector cannot see an omission. | Interpretation constraint only (audit aggregates). No causal claim. | A label-aligned, clip-centred event-triggered average over the 5 positives, compared with the same relative positions transplanted into the 12 negatives (null N1 below). If there is no difference at any lag in [−20, +5] s, there is no shared visible within-clip signature in these features. |
| D3 | **Duration and clip cutting are the strongest "signals" in this sample.** Shorter clips are positive, and relative label positions look concentrated: the permutation hit p95 of 0.6–0.8 *with* relative-position carryover implies this. Any max-over-time clip score is length-dependent (median \|ρ\| 0.33). | Duration-only 0.60 > both stage results. Synthetic: with near-end relative positions (0.7–0.95), a nested **position-prior localiser (no pixels) hits 0.98**. | Nested position-prior hit on the real labels (label-only computation, one number). **If it is ≥ 0.8, the ±5 s hit criterion cannot distinguish event detection from clip cutting on this sample**, and localisation must be tested against the prior and with the position-matched null N1. |
| D4 | **The localisation metric is weak.** ±5 s on a 24 s clip covers ~40 %. One clip moves the hit by 0.2. The strict `>` against a perm p95 of 0.8 forces 5/5. `run_end` (chosen 15/17) favours late steps, which aligns with near-end labels by construction. | Stage 5: hit 0.20 < null p50 0.40, i.e. *worse* than chance. That is consistent with D1 (scene-driven peaks), not with a weak real cue. | Report event F1 (right clip **and** right time) against the whole-procedure permutation; report hit against max(random, position prior). |
| D5 | **Power (C5-2) is limited by the procedure as well as by n = 5.** The 24-config grid with per-fold reselection and a positive-heavy F1-max threshold (C5-3: TN = 0) inflates the null. | Real perm F1 p95 0.667 vs always-positive 0.4545. Synthetic WCR (one config, no grid): null perm F1 p95 **0.58–0.60**, so the bar is lower *because the procedure is smaller*, not because α changed. | Built into Direction A (power table, §3.6). |
| D6 | **Feature semantics.** The SigLIP2 ONNX outputs only the **pooled** MAP-head embedding (`output_names=["emb"]` = `get_image_features`; no patch tokens). That is global scene gist at 224², where hands and banknotes are a few patches. The 8 prompts are guesses, and a top-view CCTV frame is out of the web distribution, so prompt cosines mostly carry per-clip offsets. YOLOX at 640 px has no cash, banknote or drawer class. Stage 4 used the 48×27 motion grid only as 3 scalars. | Stage-5 family T was chosen 8/17, E 9/17, with no stable winner. | The same η² test as D1 for prompt cosines. The within-clip SD of the cosines vs their between-clip SD (prediction: ratio < 0.3). |

Not a cause: leakage (refuted by perturbation tests in both stages) and the length bias of the max-logit score
on null data (C5-10, checked).

## 2. What existing assets allow (no downloads)

| Asset | Content (from code, not opened) | Allows |
|---|---|---|
| `local_private/train_probe/cache/<sha16>_4.npz` (17) | `t` at 4 fps; `grid` (S, 27, 48) float16 motion-fraction cells; `dets` rows (frame, x0, y0, x1, y1 normalised, score, class) for the 13 COCO classes; `dur` | Label-free zone (`estimate_zone`), **spatial motion maps inside zone and ring** (sub-grid, temporal spectrum ≤ 2 Hz), person geometry, COCO object presence. No re-decode needed. |
| `cache_emb_siglip2/<sha16>.npz` (17) | `t` at 1 fps (k + 0.5); `E_full`, `E_zone` 768-d L2 float16 **pooled**; `zone_fallback`; `dur_dec`. 656 frames, 622 valid steps | Any clip-centred / within-clip function of pooled embeddings. Re-projection onto **new text prompts** at no extraction cost. |
| `vision.onnx` (sha 4e95b080…) | Pooled output only (`emb`) | **No patch tokens.** These would need a re-export from the local `model.safetensors` in `.venv-research` (export ~2.5 s) as a new pinned artifact, plus one new real pass over train-17 (stage-5 measured: decode 5.5 s, embed 75 s for 2 views). Storage ~200 MB float16 per view. Not required for Direction A. |
| Text tower (`.venv-research`, `child_text`) | `get_text_features`, ~1 s for 8 prompts | New prompt sets can be computed cheaply. New prompts must be pre-registered from the norm text (docs/NORMS.md cues), not from data. Not used in Direction A, to avoid another guessed family. |
| YOLOX dets | persons + backpack/handbag/suitcase/bottle/cup/chair/tv/laptop/mouse/keyboard/cell phone/book | Only `cell phone` relates semantically to the norms. The rest is scene clutter (a D1 hazard). |

## 3. Direction A (recommended): within-clip contrastive ranking (WCR) — scene-free by construction

### 3.1 Hypothesis
If a labelled violation has any visible correlate in the operator zone, the correlate is a **change within the
clip** around the label second, relative to the same clip's other seconds. It is *not* a property that separates
one clip's scene from another. A detector that only ever sees within-clip deviations cannot learn scene identity
and should transfer across clips.

### 3.2 Why it addresses C5-1 and C5-2
- **C5-1.** Every feature is centred on its own clip's temporal median before anything is learned. Any additive
  per-clip offset (scene, lighting, camera, static clutter) is removed exactly.
  - The direction is learned **only from within-positive-clip differences** (label span minus the same clip's
    background).
  - Negatives contribute only a label-free scale (pooled SD).
  - So no parameter can encode "which scene is positive". The synthetic scene-only fixture gives GO 0/12 (§3.6).
- **C5-2.**
  - One configuration per feature family: no grid and no per-fold reselection. This shrinks the chance F1 p95
    from 0.667 to ~0.59 on the same n.
  - The verdict score is the max of a clip-centred, globally scaled score. It is comparable across clips without
    an F1-max-over-grid search.
  - The joint statistic (event F1) uses clip and time together.

### 3.3 Formulation (fixed now; nothing tuned)
Per clip, label-free:
- c_t = x_t − median_t(x), with x the per-step feature vector.
- z̄_t = mean of c over (t − 5, t]. A step is valid with ≥ 3 samples, as in stage 5.

Training on fold Tr:
- sd = per-dimension SD over all valid steps of Tr. This is label-free.
- For each positive k ∈ Tr: δ_k = mean z̄ over A_k = [tl − 8, tl + 2] − mean z̄ over B_k = {t < tl − 15 or
  t > tl + 7}.
  - The B term is 0 (the clip median) when |B_k| < 3.
  - A_k is chosen so that windows ending in it cover [tl − 13, tl + 2]. This tolerates a non-endpoint label of
    the T1 "middle" type.
- w = mean_k δ_k / sd: a difference of within-clip means with diagonal scaling. No λ, no PCA.

Scoring (label-free):
- s_t = (z̄_t / sd) · w.
- Clip score = max over valid t.
- **Localisation t̂ = argmax** (single rule; no peak/run_end choice).

Verdict:
- θ = `fit_thr` (F1-max, ties → higher, as before) over the 16 inner held-out clip scores. Each inner score comes
  from a direction trained on all \ {k, j}.
- Outer verdict(k) = score(k) ≥ θ, with one prediction per clip.

Feature families (pre-registered, Holm-corrected at α = 0.05 over 2):
- **A-sem:** E_zone (768-d), from the existing cache.
- **A-mot:** from the stage-4 cache, resampled to 1 s bins:
  - the zone motion 3×3 sub-grid (9);
  - ring motion and outside-ring motion (2);
  - n_in_zone, n_in_ring, zone_to_other_dist, person_area_max in the zone (4);
  - cell-phone-near (1);
  - 16-d in total.
  - Each step is then clip-centred exactly as above.

`full`-view and prompt families are **not** included: fewer attempts, and D6 says the prompts are guesses.

### 3.4 Nested validation (cannot learn scene identity)
- Outer clip-LOO over 17.
- Inner clip-LOO only for the threshold. With no config selection, the inner LOO has a single purpose.
- The direction, sd and θ never see the outer clip. The outer label and time are never read for its own score.
  The stage-5 perturbation test is repeated: 0 outer changes required, and an inner flip must change something.
- Per-clip centring is a per-clip, label-free operation that is valid at inference. There is no cross-clip
  centring.

### 3.5 Baselines, controls, GO rule
Baselines:
- always-positive;
- **duration-only nested** (0.60);
- random-time (uniform over the clip and over the support, max);
- **nested position prior** (t̂ = median training relative label position × d);
- **label-free novelty** (s_t = ‖z̄_t‖, zero parameters; reported, not a GO path);
- the stage-5 observed values for reference.

Controls:
- **N2: whole-procedure permutation, n = 1000**, seed 0. It shuffles the labels, carries relative positions over
  as in stage 5, and reruns the direction, inner thresholds and verdict.
- **N1: position-matched transplant.**
  - Each positive's fold model scores every negative at that positive's relative label position.
  - The within-clip percentile rank at the label in positives is compared with 10⁴ matched draws from negatives.
  - This rejects clip-end and edge artefacts. The precise definition is in §3.5a.
- **No circular-shift null is used anywhere.** The max of a series and the multiset of ‖e_t − median‖ are invariant
  under a circular shift, so such a null is degenerate for the clip verdict and for length calibration. Length is
  handled by criterion 6 (|ρ|) and by the nested duration-only baseline, not by shifting.

**GO** for a family only if **all** of the following hold (stage-5 criteria 1–7 kept, plus 3 new):
1. clip F1 > always-positive;
2. clip F1 > duration-only (0.60);
3. clip F1 > N2 p95;
4. hit > max(random-time);
5. hit > N2 p95;
6. |ρ(outer score, duration)| < 0.5;
7. 17/17 complete, 17 selected and 0 skipped (C5-5);
8. **event F1 > N2 p95**;
9. **N1 p < 0.05 / 2** (Holm);
10. **Position-matched localisation**, which is the localisation criterion of record:
    - (a) hit ≥ the nested position-prior hit;
    - (b) mean |t̂ − tl| over positives < the same quantity for the nested position prior.
    - (b) was **not simulated**, so its power is unmeasured.

Absolute hit thresholds (4, 5) are kept only for continuity with stage 5. They are necessary, not sufficient, and
not evidence of localisation on their own: the position prior alone reaches hit 0.98 when labels sit near the clip
end. Localisation is claimed only through 9 + 10 (detector vs position prior, position-matched).

Anything else is NO-GO. A GO licenses only a proposal of one locked holdout evaluation, after independent review.
The holdout stays closed, production stays fail-closed, and verdicts stay strictly violation | no_violation, with
missing caches counted as failed.

Optional amendment (only if approved BEFORE the run): replace criterion 5 by criterion 8. Rationale: the
5/5-hit requirement collapses power at 4/5 (below). Default = strict.

### 3.5a N1 position-matched transplant null (exact definition, for the reviewer)

**Statistic.**
- For a held-out positive k: s^(k) is the score from the fold direction w_{−k}.
- u_k is the within-clip percentile rank of s^(k) at τ_k, the valid step nearest tl_k:
  u_k = (#{valid t : s_t < s_τ} + ½(#ties − 1)) / (n_valid − 1).
- Q = mean of u_k over the 5 positives.

**What is transplanted.** Only the *label position*, never content or scores.
- For each positive k and each negative j (12 clips), j is scored with the direction w_{−{k,j}}. That is the same
  procedure as for k, with j also excluded from the label-free sd.
- u_{k,j} is computed at the transplanted time τ_{k,j}, under two pre-registered anchors:
  - **relative:** τ = (tl_k / d_k) · d_j;
  - **end-anchored:** τ = d_j − (d_k − tl_k), clipped to [0, d_j];
  - in both cases τ is snapped to the nearest valid step.
- One null draw = mean over k of u_{k,J_k}, with J_k drawn uniformly from the 12 negatives (with replacement).
- 10⁴ draws; p = (1 + #{null ≥ Q}) / (1 + 10⁴).
- **Both anchors must give p < 0.025.** The synthetic pilot measured the relative anchor only.

**Why it is non-degenerate.**
- Q depends on *where* in the series the rank is read, and on content at that place.
- The null keeps the position (and so the edge proximity, under both anchors) and the scorer fixed, while it varies
  the clip content: positive versus negative clips.
- A scorer that peaks at clip ends or at a fixed relative position gets the same u in negatives. Synthetic
  edge-artefact fixture: N1 p < 0.05 in only 1/12 reps, GO 0/12.

**Boundary effects.**
- The negatives use the identical valid-window rule (≥ 3 samples in (t − 5, t]) and the same clip-median centring.
- Two anchors are used because the negatives are longer (duration AUC 0.283). Relative matching alone would put
  the transplanted point farther from the end in seconds; end-anchored matching equalises the absolute distance to
  the end.
- Percentile ranks normalise for the different numbers of valid steps.

**Autocorrelation.**
- Nothing is resampled within a series. Each negative contributes its own intact series, so its autocorrelation
  is preserved.
- The randomisation unit is the clip.
- Limitation: only 12 negatives, reused across the 5 k's. The null granularity is coarse, and it represents these
  12 scenes only.

**What N1 does not control.** A clip-level difference that happens to co-locate with the label in positives only.
That cannot be separated from a real event on this sample.

### 3.6 Pilot: synthetic validity → timing → real cost (done for the synthetic part)
Fixture:
- 17 units, durations like the real aggregate, 5 positives drawn among the shorter clips.
- 768-d unit vectors: scene offset ‖1‖, AR drift 0.2, noise 0.35, cue in [tl − 8, tl] with per-clip variation.

Run: `wcr2.py`, 12 reps per level, n_perm 200. All levels completed; nothing is censored.

| Fixture | clip F1 (perm p95) | hit (perm p95; pos-prior) | event F1 (p95) | GO-strict | GO with crit 5→8 |
|---|---|---|---|---|---|
| null | 0.30 (0.58) | 0.40 (0.73; 1.00) | 0.15 (0.32) | **0.00** | 0.00 |
| plant 0.08, 5/5 | 0.51 | 0.97 (0.80) | 0.51 (0.37) | 0.08 | 0.17 |
| plant 0.12, 5/5 | 0.91 (0.59) | 1.00 (0.82) | 0.91 (0.42) | **0.83** | 1.00 |
| plant 0.15, 5/5 | 0.91 | 1.00 (0.83) | 0.91 (0.46) | 0.83 | 1.00 |
| plant 0.15, **4/5** | 0.77 | 0.83 (0.83) | 0.77 (0.43) | **0.08** | 0.08 |
| plant 0.15, spread positions (0.3–0.95) | 0.90 | 1.00 (0.80; prior 0.62) | 0.90 (0.40) | 1.00 | 1.00 |
| **scene-only confound** | 0.36 | 0.40 | 0.15 | **0.00** | 0.00 |
| **clip-end artefact in all clips** | 0.33 | 0.90 (1.00) | 0.31 (0.49) | **0.00** (ρ 0.81, N1 fails) | 0.00 |

Comparison with the real `stage5_model.nested` on the same generator (6 reps, no perms):
- plant 0.08 / 0.15 / 0.25 → outer F1 0.48 / 0.43 / 0.48; P(hit = 1) 0.17 / 0.33 / 0.67;
- scene-only → F1 0.91.

Caveats:
- Synthetic amplitudes are not comparable to real ones.
- The fixture has no omission-type events.
- Under GO-strict, the 4/5 result is near 0.08 under either rule. **A single clip without a cue sinks GO.**
- With near-end labels, the position prior hits 0.98–1.00, so criterion 10 (≥) is near-binding.
- The pilot differs from the spec in three ways:
  - N1 scored negatives with w_{−k} (not w_{−{k,j}}) and used the relative anchor only;
  - criterion 10b was not computed;
  - ρ was computed on outer scores (stage 5 used inner held-out scores).
- These differences belong to the self-test of the locked implementation, not to new sweeps.

Timing (measured on synthetic data of the same size as the real data, 656 steps × 768):
- one full nested WCR run: **0.034 s**;
- 1000 perms ≈ 35 s per family;
- N1 < 1 s.

Projected real train-17 cost:
- load caches < 5 s; A-mot features from the stage-4 caches < 60 s (stage-4 total was 53 s);
- 2 families × (run + 1000 perms + N1) ≈ 1.5 min;
- perturbation test < 1 min;
- the synthetic self-test (null / planted / scene / edge / 4/5 / missing-cache → failed) at n_perm 200 ≈ 8 min;
- **total ≈ 12 min wall, < 1 GB, 0 downloads, 0 new extraction**. That is well inside 60 min.

Implementation: one new `tools/stage7_wcr.py` (~250 l) plus a self-test, reusing `stage4_probe.fit_thr/clip_f1/
spearman`, `stage5_model.windows/wmean/permute/hit_rate` and `train_probe.features`.

## 4. Direction B (conditional, not recommended for tonight): zone patch-token novelty
- **Hypothesis.** Small hand or cash events change a few zone patches. Pooling hides them (D6).
- **Formulation.**
  - Re-export the vision tower with `last_hidden_state` (196 × 768), as a new pinned ONNX artifact with a parity
    check.
  - One local train-17 pass for the zone view.
  - Score: s_t = top-5 mean over the zone's patches of ‖p_t,i − median_t(p_i)‖, which is per-patch clip-centred
    novelty. **Zero learned parameters**, so scene identity cannot be learned and there is no selection.
  - Evaluate with exactly the §3.5 rule; the threshold is still nested.
- **Cost.** Export about 1 min; real pass about 2 min (stage-5 measured); ~200 MB local cache; scoring and perms
  < 2 min.
- **Risk.** A new derived artifact, plus a real extraction pass that needs coordinator approval.
- **Attempts.** It is a third attempt: if it is added to the same pre-registration, α becomes 0.05/3 (Holm).
  Otherwise it is a new stage. It must not be run *because* A failed without that correction.

## 5. Honest expected outcome and NO-GO reporting

**Expectation.**
- D1 is very likely real (the synthetic scene-only result plus the real fold instability). WCR removes it, so
  stage-7 numbers should no longer sit *below* the null median.
- A GO still needs a cue that is strong and present in **all 5** positives. One unseen or omission positive
  (V1 4/5 unseen) is enough to fail criteria 5 and 10.
- Criterion 2 (beat duration-only 0.60) and criterion 10 (beat a position prior that may be ~1.0) are the likely
  binding failures.
- **Prior: NO-GO is more likely than GO.** "~70/30 at best" is a **subjective** judgement, not a measured
  probability. The only measured quantities are the synthetic GO rates in §3.6.

**If NO-GO, report exactly:**
- per family: all 10 criteria with observed vs bar, both nulls (p50 / p95 / p_perm), and the position-prior and
  novelty baselines;
- the D1 η² and D3 position-prior aggregates as diagnostics.

The statement to make: "a scene-free within-clip detector with fixed design did not show a transferable
within-clip signature on train-17 (5 positives); with the cue in 4/5 positives the synthetic power is ≈ 0.08, so
this does not show absence of signal; the binding limits are n = 5, label endpoints of unknown visibility, and
clip-cutting structure (duration, relative position)."

In every case: production stays `failed/no_validated_rule`, the holdout stays closed, and nothing is enabled.
The disclosure carries over: adaptive exploratory research, third attempt on the same sample, audit-informed,
holdout seen once in stage 1.
