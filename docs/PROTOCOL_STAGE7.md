# PROTOCOL STAGE 7: within-clip contrast against unlabeled background (WCR), two feature families (pre-registration, LOCKED)

**Status: LOCKED 2026-10-18 02:35 UTC** (pre-lock reviewer revisions R1–R6 applied; R5 output-only diagnostics added to the code;
S7-1 stricter N1 pool; GO bars unchanged). The real run waits for the reviewer to confirm the text diff. Before the lock, the only computations were on synthetic fixtures (§11). After
the lock, any change is a dated amendment appended at the end. After the real run, §1–§9 never change.

Implementation:
- `tools/stage7_wcr.py`: loader, child process with watchdog, CLI and summary.
- `tools/stage7_core.py`: the numeric core.
- `tools/stage7_selftest.py`: the synthetic self-test.

**Number of attempts: 1.** This protocol is the single stage-7 attempt.

## 0. Disclosure and scope (binding)
- This is the **third adaptive look at the same train sample** (C02 train, n = 17, 5 positives), after stages 4
  and 5. It is exploratory, and no independent validation is claimed.
- The design was informed by:
  - the human audit aggregates (T1 4 end / 1 middle; V1 4 unseen / 1 partial);
  - the stage-4/5 results and reviews (C5-1 scene identity, C5-2 power, C5-3 threshold);
  - STAGE7_DIAGNOSIS_ARCH.md.
- The holdout was seen once in stage 1. It is not opened here, and it stays closed even under GO until a separate
  agreement.
- Production stays fail-closed (`failed/no_validated_rule`). No normative rule is enabled.
- Verdicts are strictly `violation | no_violation`. A missing or invalid cache counts as a failed unit, never as
  `no_violation`.
- **Data.**
  - Only the existing numeric caches are read: stage-4 `train_probe/cache/<sha16>_4.npz` and stage-5
    `train_probe/cache_emb_siglip2/<sha16>.npz`.
  - Units come from `select_split_units(LP, cid, "train", dry_run=True)`, so **no clip file is opened**.
  - There is no new extraction and no download.
- **Outputs.** stdout and `train_probe/stage7_summary.json` (mode 0600) carry aggregates only: no shas, paths, names,
  per-clip rows or per-clip times.
- **Immutables.** sha256 before and after for `labels/*`, `inventory_map.json`, `stage4_summary.json` and
  `stage5_summary.json`. Only `unchanged` or `CHANGED` is printed.
- **Localisation target.** The timestamp endpoint: the self-chosen time within ±5 s of the imported label second.
  It is not an onset, an interval or a legal violation.
- **All success probabilities in this document are subjective unless they are explicitly labelled as synthetic
  measurements.**

## 1. Hypothesis
- If a labelled violation has a visible correlate in the operator zone, it shows as a **change within the clip**
  around the label second, relative to the same clip's other seconds.
- WCR learns only from such within-clip differences between the label window and the **unlabeled** background of
  the same clip. It is designed to reduce the stage-5 failure mode (C5-1),
  where 5 positive scenes were separated from 12 negative scenes.

**LIMITATION (not a property).**
- Clip-median centring removes an additive static per-clip shift.
- It does **not** guarantee that scene identity cannot be learned. With normalised, non-linear embeddings, per-clip
  scale, dynamics and heteroskedasticity remain. A busier scene has larger within-clip deviations, and the shared
  direction can align with scene-specific dynamics.
- Partial controls:
  - the label-free pre-check of the clip-id share of the centred-step-norm variance (§8);
  - the position-matched transplant null N1 (§6);
  - the synthetic scene-only and clip-end fixtures (§11).
- None of these is a proof.

## 2. Feature families (fixed; Bonferroni over 2, §7)
Every family is first resampled to 1 s steps (t = k + 0.5) and then processed per clip, label-free:
- **centring:** c_t = x_t − median_t(x);
- **window:** z̄_t = mean of c over (t − 5 s, t]. A step is valid with ≥ 3 samples.

- **A_sem.**
  - SigLIP2 B/16-224 zone view, `E_zone`: 768-d, L2-renormalised, from the stage-5 cache at 1 fps.
  - Zone fallback (full frame) is counted, never excluded.
- **A_mot.** 16-d, from the stage-4 cache at 4 fps, averaged into 1 s bins [k, k+1):
  - 9 dims: the motion fraction of the 3×3 sub-grid of the label-free operator zone (`train_probe.features` →
    `estimate_zone`; the 48×27 grid cells whose centre lies in each sub-rectangle);
  - `motion_ring`, `motion_global`, `n_in_zone`, `n_in_ring`, `zone_to_other_dist`, `person_area_max`,
    `phone_near` (as in `train_probe.features`).
  - If the zone is unavailable, the unit has no A_mot features. It is counted as failed for A_mot, and A_mot
    cannot GO (§7, criterion 7).

Not included: the full view, the text-prompt family (guessed prompts, D6) and PCA (it would re-introduce
between-clip components).

## 3. Within-clip contrast target, direction, scores
**Target, positive clips only.** For label time tl:
- A = valid steps with t ∈ [tl − 10, tl + 2]. Their windows cover raw time [tl − 15, tl + 2].
- B = valid steps with t < tl − 15 or t > tl + 7. This is the **unlabeled** same-clip background; the guard band guarantees that
  no B window overlaps the raw span of A.
- δ_k = mean(z̄ over A) − mean(z̄ over B). If |B| < 3, the B term is 0, i.e. the clip median, which is the reference
  after centring. If A is empty, the clip gives no δ.
- Justification:
  - T1 says the labels are mostly endpoints, so the evidence should precede the label.
  - The +2 s and the 15 s raw reach tolerate one "middle" label and late annotation.
  - The span equals stage 5's, so the target is unchanged and only the contrast is new.
- **Negative clips contribute no target**, only steps to the label-free scale. Their clip-level
  `reviewed_no_violation` status is an imported XLSX label that no human here has verified.

**Background assumption: UNLABELED, not verified negative.**
- The label timestamp usually marks the *end* of an event. It does not certify that the rest of a positive clip is
  free of the event. B is therefore unlabeled background used only for contrast. It is not a set of supervised
  negative windows, and the background clips are not "pairs".
- What supports the assumption locally (safe aggregates only):
  - T1 (4 end / 1 middle): the labels mark endpoints, so the evidence is more likely *before* the label. That is the
    reason for the 15 s reach and the guard band. T1 says nothing about whether the event recurs later.
  - V1 (4 unseen / 1 partial): the audit could not see the behaviour. This gives **no** support that anything is
    visible.
  - The 12 negatives' whole-clip `reviewed_no_violation` statuses are imported XLSX clip labels and are not
    human-verified here.
- **Everything else about the background is UNVERIFIED.**
- Consequences for the statistic:
  - If the event extends before tl − 15 or recurs after tl + 7, those steps are counted as background. δ_k shrinks
    and the within-clip rank u_k falls.
  - If the event covers more than half of a short clip, the clip median itself contains event steps and the
    centred contrast shrinks.
  - Event contamination inside B costs power **under the alternative only**. Systematic non-event structure in B
    is UNKNOWN on real data.
- Validity conditions of the two nulls (stated exactly; neither condition is verified):
  - **N2** (whole-procedure permutation) is exact only under full exchangeability of (status, relative label
    position) across the 17 sequences. That is **known to be false here**: duration AUC 0.283, and ±5 s covers
    different shares of short and long clips. An N2 rejection therefore means *some* label–feature association,
    which may include duration or clip cutting. It does not mean event signal.
  - **N1** (transplant) is valid only if positives and negatives share the same position-dependent rank structure at
    the matched anchor. This is **UNVERIFIED**:
    - the negatives are unpaired and longer;
    - neither anchor matches the distance to the clip start;
    - a positive-specific change co-located with the label is not excluded. It could come from clip cutting, from
      the annotator's choice of endpoint, or from a scene change that is not a violation.
- **UNVERIFIED controls limit interpretation even under a numeric GO.**
- Approach check, made now before any real result: no change. A narrower background (before the label only) would
  leave too few steps in short clips. The |B| < 3 fallback to the clip median is kept.

**Scale (label-free, robust).**
- sd_d = max(1.4826 · MAD_d, 0.1 · SD_d, 1e-6), computed over all valid steps of the fold's training clips.
- It is cached by training set. It never sees labels, so permutations reuse it unchanged.
- MAD is used because the cue steps of positives (~10 % of the steps) inflate the SD and shift the score scale
  between fits (synthetic finding, §11).

**Direction (fixed, no grid).**
- w = normalise( mean_k δ_k / sd ) over the fold's training positives. Unit L2 norm; zero if there is no δ.
- This is a diagonal-LDA-type difference of within-clip means. Its regularisation is the robust diagonal scale, the
  averaging over positives and the unit norm. There is no λ, no PCA and no hyper-grid.

**Per-step score.** s_t = (z̄_t / sd) · w on valid steps.

**Clip score.** max over valid t.
- **Length-dependence mitigation: none, by design.** A max grows with the number of steps, which favours long clips.
  In this sample the long clips are the negatives (duration AUC 0.283), so the bias works *against* the known
  "shorter = violation" shortcut.
- Any length normalisation (for example dividing by √(2 ln n)) would raise the scores of short clips and align the
  detector with that shortcut. Circular-shift calibration is degenerate for a max and is not used.
- The residual dependence is gated by criterion 6.
- ρ expectation (synthetic self-test, median |ρ| over folds):
  - null 0.46 (A_sem) / 0.27 (A_mot);
  - scene-only 0.55 / 0.33;
  - clip-end artefact 0.61 / 0.59;
  - planted 0.09–0.15.
- On real data criterion 6 may bind. That is accepted.
- Criterion 6 is **unsigned**. The low planted |ρ| comes from cancellation (the positives are short and score high),
  not from an absence of length dependence. In the reviewer's null fixture, a median |ρ| of 0.46 passed the bar in
  41/60 seeds; this protocol's own 60-seed null rate is in §11.
- Diagnostics, reported with no bar (approved by the coordinator before the lock):
  - the signed median over folds of ρ(inner held-out scores, durations);
  - the same median, computed over the negative clips of each fold only.
  - They appear in the summary as `DIAGNOSTIC_signed_rho_median_all_and_negatives_only` and are printed on stdout.
  - **No criterion, threshold or selection reads them** (`stage7_core.py` lines 163, 179–180, 316–318).

**Localisation.** t̂ = the earliest argmax of s_t (`peak`). This is fixed: there is no run_end and no choice.

## 4. Nested clip-LOO (17 outer × 16 inner); fold-wise sets written out
For outer clip k (all = the 17 units of the family):

| Object | Fitted on (labels used) | Scored out-of-sample |
|---|---|---|
| outer scale sd_{−k} | steps of all \ {k} (no labels) | — |
| outer direction w_{−k} | δ of positives in all \ {k} | clip k (score, t̂, within-clip rank) |
| pair scale and direction w_{−{k,j}} | all \ {k, j} | clip j (inner held-out score) |
| threshold θ_k | `fit_thr_mid` over {s_{−{k,j}}(j) : j ≠ k}, labels of j ≠ k | applied to s_{−k}(k) |

- Outer verdict: s_{−k}(k) ≥ θ_k. **There is one prediction per clip.** Clip k's label and time are never read
  for its own prediction.
- **Negatives in N1 for positive k are scored with w_{−{k,j}}.** So negative j takes no part in the direction or
  the scale that scores it, and k takes no part either (this closes the spec gap noted in the diagnosis).
**Strict clip-wise partition (every fitted object).**
- No clip is ever in the fitting set of an object that scores it. This covers the scale, the direction and the
  threshold, and holds for the primary run, every N2 replicate and N1.
- The only per-clip operations that use the scored clip itself are its own median centring and its own window
  means. These are label-free transforms of that clip alone and are available at inference.
- PCA is not used.
- The §8 pre-check pools all clips, but it is label-free, reported only, and never feeds a decision.
- Fold-wise sets for outer fold k:
  - **fit(direction_out, scale_out)** = all \ {k}; **score** = {k}.
  - **fit(direction_pair_j, scale_pair_j)** = all \ {k, j}; **score** = {j}, for each j ≠ k.
  - **fit(θ_k)** = the inner held-out scores of {j ≠ k} with their labels; **apply** = {k}.
  - **N2 replicate**: identical sets, with permuted labels and positions.
  - **N1 for positive k**:
    - the observed u_k is scored by the {k}-excluded model;
    - each negative j is scored by the {k, j}-excluded model. The negatives scored for N1 never enter the scale or
      the direction that scores them.
- `fit_thr_mid` chooses the F1-max threshold among the midpoints of adjacent sorted training scores (and min − 1).
  Ties go to the higher threshold. This replaces stage 4/5's cut at a training score, which leaves zero margin
  (synthetic finding, §11; it relates to C5-3).
- Perturbation test (self-test): every outer label flip or time shift must cause 0 changes of that clip's score,
  t̂ and verdict, and an inner flip must change other clips.

## 5. Metrics and baselines (per family, outer folds only)
Metrics:
- **Clip level:** TP / FP / FN / TN and F1.
- **Hit:** the share of positives with |t̂ − tl| ≤ 5 s, regardless of verdict.
- **Event F1** (±5 s): a predicted-violation clip is a TP when it hits. F1 = 2TP / (2TP + FP + FN), with FN
  counted over label times. It is cross-checked against `evaluate.match_events`.
- **MAE:** the mean over positives of |t̂ − tl|.

Baselines:
- **Always-positive** F1.
- **Duration-only nested**, exactly as in stages 4/5 (the `stage4_probe.nested` sign and threshold). Stage-5 value:
  0.60.
- **Random time**, the max of uniform over [0, d] and uniform over valid steps.
- **Position priors** (nested, label-free at inference), both anchors:
  - relative: t̂ = median_{training positives}(tl/d) · d_k;
  - end-anchored: t̂ = d_k − median(d − tl).
  - MAE and hit are reported for each.

## 6. Nulls
**N2: whole-procedure permutation.**
- n = 1000, seed 0 (A_sem) and 1 (A_mot).
- Labels are shuffled over the 17 clips, and the relative label positions are carried over to the receiving clip's
  duration, as in stages 4/5.
- Each replicate reruns §4 completely: δ for the pseudo-positives, all directions, inner thresholds and verdicts.
  The fold-wise sets are the table in §4. Only the label-free scales are reused.
- p_perm = (1 + #{null ≥ obs}) / (1 + n), computed for clip F1, hit and event F1.

**N1: position-matched transplant null** (no circular shifts are used anywhere).
- **Statistic.**
  - u_k is the within-clip percentile of s_{−k} at the valid step nearest tl_k, among all valid steps of clip k:
    (#{s < s_τ} + ½(#ties − 1)) / (n_valid − 1).
  - Q = mean over the 5 positives.
- **Transplanted object.** Only the *label position* is transplanted, never content or scores.
  - For each positive k and each of the 12 negatives j, compute u_{k,j} at τ_{k,j} on s_{−{k,j}}(j), with two
    anchors:
    - relative: τ = (tl_k / d_k) · d_j;
    - end-anchored: τ = max(0, d_j − (d_k − tl_k)).
  - τ is snapped to the nearest valid step.
- **Null draw** (amended before the lock, S7-1).
  - For each positive k, the reference pool is {u_{k,j} : the 12 negatives j} ∪ {u_k}: 13 values, with the
    observed u_k included in its own pool.
  - A draw = mean_k of one value per k, drawn uniformly from its 13-value pool (with replacement). 10⁴ draws.
  - p = (1 + #{null ≥ Q}) / (1 + 10⁴).
  - Reason: with a 12-value pool the null was anti-conservative. The reviewer measured a false-rejection rate of
    0.044 at nominal 0.025, and the variance-ratio argument gives (1/5 + 1/12)/(1/5) ≈ 1.42. Including the observed
    value is stricter and lowers no bar.
  - The non-degeneracy check (pool SD > 0 and > 50 distinct null values) still uses the negatives' values.
- **Fold-wise separation.**
  - u_k uses a model in which k is out-of-sample.
  - u_{k,j} uses a model in which both j and k are out-of-sample.
  - So no negative that is scored for the null helped to fit the direction or the scale that scores it.
- **Non-degeneracy.**
  - Q reads the rank at a specific position, so it is not shift-invariant.
  - The null keeps the scoring procedure and the matched position under one anchor at a time: the relative
    position (relative anchor) or the distance to the clip end (end anchor). Only the end anchor keeps the distance
    to the end. **Neither anchor keeps the distance to the clip start.**
  - It varies whether the clip content comes from a positive or a negative clip.
  - The self-test asserts that the pool SD is > 0 and that there are > 50 distinct null values for both anchors.
- **Boundary effects.**
  - The negatives use the identical window-validity rule and the same median centring.
  - Two anchors are used because the negatives are longer: relative matching equalises the proportional position,
    end-anchored matching equalises the absolute distance to the end. Neither equalises the distance to the start.
  - Percentile ranks normalise for n_valid. **Both anchors must pass.**
- **Autocorrelation.** Each negative's series is used intact, never resampled within a clip. The randomisation unit
  is the clip.
- **Limits.**
  - Only 12 negatives, reused across k, so the null is coarse.
  - The edge distance is matched only partially: the end anchor matches the distance to the end, the relative anchor
    matches neither absolute edge distance, and neither matches the distance to the start. The window-validity rule
    and the median centring make ranks near the clip start and end systematically different.
  - **First label only.** N1 and the position priors use only the first label of each positive
    (`stage7_core.py` lines 212, 221, 243–246). The real positives carry one label each, but a second label would
    be ignored there.
  - A label-co-located difference that is specific to positive clips cannot be separated from a real event.

## 7. Decision rule, multiplicity
**Multiplicity: Bonferroni over the 2 families.** α_f = 0.05 / 2 = 0.025 for every p-value criterion of a family.

A family passes only if **all 10** of the criteria below hold. The criteria form an intersection–union test, so no
further within-family correction is applied.

**Overall GO** = at least one family passes. Otherwise the result is a measured **NO-GO**.

| # | Criterion (same bars as stage 5 where they exist; p-value bars stricter) |
|---|---|
| 1 | clip F1 > always-positive F1 |
| 2 | clip F1 > duration-only nested F1 |
| 3 | N2 p_perm(clip F1) ≤ 0.025 (stage 5: > p95) |
| 4 | hit > max(random-time uniform, random-time valid support) |
| 5 | N2 p_perm(hit) ≤ 0.025 |
| 6 | median over folds of \|Spearman(inner held-out clip scores, durations)\| < 0.5 |
| 7 | family complete: 17 labelled units selected, 0 skipped, 0 failed for this family |
| 8 | N2 p_perm(event F1) ≤ 0.025 |
| 9 | N1 p ≤ 0.025 for **both** anchors |
| 10 | MAE(detector) < min(MAE(relative prior), MAE(end prior)): **position-matched localisation** |

Notes:
- **Localisation evidence is criteria 9 and 10.** These compare the detector with the position prior at matched
  positions.
- Criteria 5 and 8 alone are **not** valid localisation tests, because N2 assumes duration exchangeability (§3).
- What a GO means: "a label-co-located within-clip difference, specific to labelled clips, beyond what N1 and N2
  produce". It does not mean detection, and it does not mean the labels are correct.
- The absolute hit criteria 4 and 5 are kept for continuity. They are not sufficient: with near-end labels the
  prior alone hits ~1.0, and then criterion 5 is structurally unpassable (§11 `nearend`).
- A failed criterion is reported as such; there is no rescue.
- If the permutations are incomplete (global deadline), criteria 3, 5 and 8 are 0, i.e. NO-GO by rule.
- **A technical PASS (self-test or any criterion) is not evidence of detection or of label correctness.**
- A GO licenses only a proposal of one locked holdout evaluation, after independent review.

## 8. Label-free pre-check (reported only; it does not enter the decision)
The clip-id share of variance (η², the between-clip sum of squares over the total), as aggregate % (median / max over
columns):
- PCA-8 of E_zone and E_full, fitted on all steps of all units, label-free;
- the 8 stage-5 prompt cosines (`text_probes.npy`, pinned sha256-16 e5c2ba8d958bfc2a; skipped if it does not
  match);
- the raw A_mot features;
- the norm of the centred steps per clip (heteroskedasticity left after centring, §1 limitation).

Prediction (D1 and D6, subjective): PCA-8 and prompt η² above 80 %. If the centred-norm η² is high, the §1 limitation
is material.

## 9. Budget, memory, fail-closed
- One child process (`--child`; job via stdin; stderr DEVNULL; stdout one JSON line of aggregates).
- The RSS watchdog follows the stage-5 pattern: the process group is polled every 0.25 s and killed above
  `--mem-mb` (default 4096) or the wall budget. The self-test proves the kill on the real entry point.
- Global wall budget: `--budget-total-sec 1800`. The permutation loop stops before the deadline; an incomplete
  permutation set means NO-GO by rule.
- Measured synthetic (17 units, same size as the real data):
  - ~7 s per scenario at n_perm 200 for both families;
  - child RSS ~120 MB;
  - full self-test 140 s.
- Projected real run: ≈ 1 min at n_perm 1000. Self-test plus real run is well under 30 min.
- Fail-closed:
  - A missing or invalid cache, or a missing zone, is a failed unit for that family (counted).
  - With < 6 units or a single class, the family is refused (family_GO 0).
  - A child failure gives exit 2 and no summary.

## 10. NO-GO report template
"Stage 7 (single pre-registered attempt, exploratory, third look at train-17 with 5 positives): within-clip
contrast against unlabeled background, with fixed design.
- Per family: clip F1 X (always-positive A, duration-only B, N2 p P); hit H (random R, N2 p P); event F1 E (N2 p P);
  MAE M s vs position prior Mr / Me s; N1 p rel / end; median |ρ| Z.
- Failed criteria: […]. Measured NO-GO.
- Synthetic GO counts at α 0.025 (Clopper–Pearson 95 % CI):
  - cue in 5/5 positives: 3/5 [0.15, 0.95] at both amplitudes;
  - cue in 4/5: 1/5 [0.005, 0.72];
  - labels near the clip end: 0/1;
  - replicated null false-GO: see §11.
  This does not show absence of a visible signal.
- Binding limits: n = 5; endpoint labels of unknown visibility (V1 4/5 unseen); clip-cutting structure (duration,
  label position).
- Pre-check η²: …
- Production fail-closed; holdout closed; nothing enabled."

## 11. Synthetic self-test (`python -m tools.stage7_selftest DIR --n-perm 200 --power-reps 5`)
**Fixture.**
- 17 train units and 4 holdout dummy clip files (chmod 000, never opened).
- Synthetic numeric caches written directly:
  - A_sem: 768-d unit vectors with a per-clip scene offset of norm 1, AR(0.9) drift 0.2 and noise 0.35;
  - A_mot: a 4 fps motion grid with a per-clip scene level and an operator box.
- Durations: 15 × U(17, 32) s plus 60 s and 142 s. The 5 positives are drawn among the 10 shortest.
- Labels: U(0.3, 0.95) · d (`nearend`: U(0.7, 0.95)).
- Cue: [tl − 8, tl], amplitude 0.15 (A_sem) and +0.35 motion in a zone sub-block (A_mot).

**Scenarios.** The result of the run before the threshold and scale fixes is not counted; the table is the
post-fix run, aggregates only.

| Scenario | Expected | Result |
|---|---|---|
| planted | GO (A_sem GO) | A_sem GO (F1 0.91, hit 1.0, all p ≤ 0.005); A_mot fails only criterion 10 (plateau peak 3 s before tl) |
| null | NO-GO | NO-GO |
| scene-only (static shift in positives) | NO-GO | NO-GO (N1 p 0.62 / 0.63 after S7-1) |
| clip-end artefact in every clip | NO-GO | NO-GO (N1 p 0.93 / 0.68 after S7-1; ρ 0.61) |
| cue in 4/5 | reported | NO-GO (criteria 5, 9, 10) |
| missing embedding cache (1 unit) | A_sem no GO | criterion 7 = 0; A_sem no GO |
| nearend labels | NO-GO (not identifiable) | NO-GO (criteria 5, 10) despite F1 1.0 |
| open() spy, in-process compute | 0 holdout opens | 0 |
| perturbation (34 per family) | 0 own changes | 0; a positive flip changed 16 others |
| memory watchdog (cap 150 MB, 300 MB alloc) | memory_cap | memory_cap |
| N1 non-degenerate, event-F1 check | pass | pass |

**Bounded power** (A_sem only, 5 reps each, n_perm 200, α 0.025; synthetic, not evidence about real data):

| Cue amplitude | Positives with cue | GO count (95 % CI, Clopper–Pearson) |
|---|---|---|
| 0.12 | 5/5 | 3/5 [0.15, 0.95] |
| 0.15 | 5/5 | 3/5 [0.15, 0.95] |
| 0.15 | 4/5 | 1/5 [0.005, 0.72] |

**Replicated null false-GO rate** (null fixture, both families, n_perm 200, α 0.025 per family, overall GO =
either family): **0/60 seeds** (95 % CI [0, 0.060]).
- Per family: A_sem 0/60, A_mot 0/60.
- Criterion 6 passed in 51/60 seeds (A_sem) and 45/60 (A_mot), with median |ρ| 0.36 for both. So criterion 6
  rarely binds under this null.
- Computed with the unchanged code (`stage7_wcr.compute` on fixture seeds 1000–1059), synthetic only.
- It was computed before the S7-1 N1 amendment. S7-1 can only remove GOs, so 0/60 remains an upper-bound-valid
  figure.
- Post-S7-1 self-test: 10/10 PASS at n_perm 100 (26 s); every scenario outcome is unchanged.

**Changes from STAGE7_DIAGNOSIS_ARCH §3.3, made on synthetic data only, before any real stage-7 computation:**
- (a) a robust MAD scale plus a unit-norm direction (an inner/outer score-scale mismatch was found);
- (b) the midpoint F1-max threshold (a zero-margin cut was found);
- (c) A = [−10, +2] (coordinator) instead of [−8, +2];
- (d) both N1 anchors and pair models for the negatives.

**Prior (subjective):** NO-GO is more likely than GO.
