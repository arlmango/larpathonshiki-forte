# STAGE 7 diagnosis: why stages 4 and 5 failed (independent critical review)

Author: critical-reviewer (stage 7), self-reported model claude-opus-5-5 (user to verify). Read-only review.
This is the only file written. Written 2026-10-18 (UTC).

**Data boundary.** No video, frame, label row, audit answer, per-clip row, sha or path was opened or quoted.
The inputs were code, docs, and the aggregate fields of `stage4_summary.json` / `stage5_summary.json`.
Experiments are numpy-only on synthetic pseudo-embeddings in the session scratchpad `diag7/`: `power.py`,
`scene.py`, `scene2.py`. They use no project data and no `tools.*` entry point.

Inputs read: COORDINATION (stages 2 to 7), PROTOCOL_STAGE4/5, REVIEW_CRITICAL (S4-x, C5-1 to C5-10), QUALITY §3–3e,
NORMS §3, tools/stage4_probe.py, stage5_model.py, stage5_probe.py, stage5_encoder.py, train_probe.py.

## 0. Bottom line
1. **The primary cause is information-theoretic, not a bug.** Train has 5 positives. The pre-registered bars need
   a near-oracle cue:
   - clip level: ≥ 4/5 TP with ≤ 1/12 FP, or 5/5 TP with ≤ 3/12 FP;
   - localisation (stage 5): 5/5 hits within ±5 s on ~24 s clips.

   Both stages also *learned* the cue direction or feature from those 5 clips. In synthetic data with the same
   shape, a *known* cue direction gives AUC 0.92–1.00 and hit 0.99–1.00. The same cue *learned* from 5 clips gives
   AUC 0.54–0.77 (claim G2). Learning, not representation, is the bottleneck.
2. **Stage 5 adds a design flaw: no within-clip contrast (C5-1 confirmed by mechanism).**
   - The positive targets come from 5 clips, and the negatives are all steps of 12 other clips (`stage5_model.py:105-118`).
   - The "level" features and the PCA fitted on all steps (`:55-58`, `:72-76`) let the model separate scenes.
   - In synthetic data, scene variance σ_s = 3 cuts the clip AUC of this formulation from 0.93 to 0.59 at a strong
     cue, while localisation survives (hit 0.96 → 0.88).
3. **The metric and decision machinery is biased against weak signals.**
   - An F1-max threshold under no signal collapses toward always-positive (real stage 5: TN = 0).
   - Selection over 24 or 224 configs, done inside 16-clip folds, shrinks a real effect further.
   - A single locked score tested by exact Mann–Whitney would need AUC ≥ 0.783 (α 0.05). The nested-F1 route needs
     ≈ 0.93+.
4. **Clip duration is the strongest "signal" in train, and it is an artefact.**
   - Shorter = violation in 17/17 folds; duration-only nested F1 0.600; duration AUC 0.717 (= 1 − 0.283).
   - It does not reach the exact p < 0.05 bar itself (AUC 0.717 < 0.783), so it is fragile too.
   - Every max-over-time score is length-dependent.
5. **No direction can produce strong evidence on train-17 before 07:00 UTC.** The best attainable result for one
   pre-registered within-clip hypothesis is p ≈ 0.01–0.03, and only if all 5 positives agree. The 16 positives in the
   holdout are the only lever that changes this materially, and they are frozen (user decision only).

## 1. Testable claims (each: claim → evidence → status → how to falsify)

### (a) C5-1 scene identity. SUPPORTED by mechanism; CONSISTENT with the aggregates
- **Code.** `targets()` (`stage5_model.py:105-118`) builds these training rows:
  - positive clips: only the rows with t ∈ [tl−10, tl+2];
  - negative clips: every valid row;
  - "Other steps of positive clips are UNUSED".

  So **no row ever contrasts a label window with the rest of the same clip.** The classifier's only task is
  "steps from positive clips vs steps from negative clips", which is a between-clip problem.
- **Features.**
  - `temporal()` (`:55-58`) concatenates `level` (the absolute window mean of z) with the residual (level − clip
    median).
  - For family E, z = PCA₈ fitted on all training steps (`:72-76`). The leading PCs of all steps over 16 clips are
    dominated by between-clip variance: camera, lighting, counter layout.
  - The residual half is within-clip, but `level` lets the model use scene identity whenever that reduces training
    loss. With 5 positive scenes it always does.
- **Synthetic mechanism** (`scene2.py`).
  - Setup: 17 clips, 32-d AR(1) pseudo-embeddings, shared distractor bursts in all clips, a cue in the 8 s before
    the label in positives, positives shorter.
  - Formulation A (stage-5 targets, level + residual) at cue a = 5: clip AUC 0.93 at σ_s = 0, but **0.59 at
    σ_s = 3**. Hit 0.96 → 0.88.
  - Formulation C (within-clip targets, residual only): AUC 0.65 at both σ_s values. It is immune to scene variance.
- **Aggregate consistency.**
  - The selection is unstable: 9 distinct configs over 17 folds.
  - F1 0.300 is below the null median of 0.375.
  - Real TP3/FP12/TN0.
- **Falsify (cheap, label-free).** On the 17 SigLIP caches, compute the share of PCA-8 variance that is
  between-clip, as an ICC per component.
  - If the between-clip share is ≤ 50 % for the top components, C5-1 is weakened.
  - Report only the aggregate ICC median and max. This uses no labels, so it re-tests no answers.

### (b) What a within-clip formulation would change. PARTLY SUPPORTED
These formulations remove the scene and duration-of-scene confounds from **localisation**:
- per-clip standardisation;
- the residual-to-clip-median only;
- label window vs same-clip rest;
- logistic with clip fixed effects (equivalent to per-clip centring);
- pairwise ranking inside clips.

They do **not** make a clip-level verdict work by themselves:
- **Clip AUC stays modest.** The synthetic C result is 0.48–0.65 at a ≤ 5, because the direction is still learned
  from 5 clips in D dims.
- **The max-of-residual score correlates with length** (synthetic r(score, duration) 0.26–0.43). Every clip has
  transients, and longer clips have more of them.
- **Localisation is the part that improves.** The synthetic within-clip rank is 0.62–0.81.
- **Consequence.** A within-clip formulation fits a **localisation-only hypothesis on the 5 positives**. A
  clip-level verdict still needs a length-calibrated score (direction D3) and remains power-starved.
- **Caveat.** Pairwise and fixed-effects forms have the same 5 independent units. The ~12 steps per window are
  autocorrelated, so they are not 60 samples.

### (c) The train set: 5 positives, labels mostly endpoints, events mostly unseen. CONSTRAINT, not cause
These are interpretation constraints only. They establish neither a cause nor invisibility.
- **T1: 4 end / 1 middle.** Window placement:
  - stage 4 ends its windows at or slightly before the label;
  - stage 5 uses [−10, +2] s.

  Both are compatible with "label ≈ event end", so placement is not the obvious failure.
- **Label position.** Preceding-window coverage is 15 s for 4/5 positives, with median duration 23.8 s, so labels
  tend to sit in the last third of the clip.
  - Any late-biased localiser (stage 5 chose `run_end` in 15/17 folds) gets credit from a **position prior** that
    no protocol baselined (C5-4).
- **V1: 4 unseen / 1 partial.** The annotator could not see the labelled behaviour at the label second in 4/5
  positives.
  - If true for the encoder too, no frame-level model can recover it; this caps every visual direction.
  - It is **not** evidence that a cue at another time (before the end-label) is absent.
- **Split imbalance (fact from QUALITY §3).**
  - Train base rate 5/17 = 29 %, holdout 16/23 = 70 %, full set 22/42 = 52 %.
  - The design choice to freeze the holdout left 23 % of the positives for learning.

### (d) The duration confound. SUPPORTED; partly avoidable
- **Facts.**
  - Shorter = violation in 17/17 folds of both stages.
  - Duration-only nested F1 0.600 (TP3 FP2), duration AUC 0.717, point-biserial r −0.187. The weak r is pulled by
    the 142.5 s outlier.
  - Chosen-score |ρ| with duration: median 0.33 and max 0.59 in stage 5; 0.18 and 0.29 in stage 4.
- **Mechanism.**
  - A max over valid steps (`stage5_model.py:131-140`; stage 4 `scan`, `stage4_probe.py:50-71`) grows with the
    number of steps.
  - The F1-max threshold then flags long negatives.
  - C5-10 found no material bias on null synthetic data. On the real run, the signed direction is unknown; |ρ|
    0.33 is not negligible.
- **Score families that avoid it.**
  1. **Within-clip localisation metrics** (rank of the label window among the clip's own steps). These are fully
     immune: no cross-clip comparison.
  2. **A per-clip calibrated max.** Compare the observed max to its own circular-shift or block-bootstrap null
     within the clip; the clip score is that p-value. The expected value no longer grows with length.
  3. **Mean-type scores** (share of steps above a fold-fitted level). These are length-invariant in expectation
     but dilute short cues.
  4. **Fixed-length scoring** (top-k mean with fixed k, e.g. k = 5 s). This only partly helps.
- Any stage-7 protocol should pre-register one of (1) or (2) and keep the "beat duration-only" bar.

### (e) Localisation on short clips. SUPPORTED: the binary ±5 s hit is a weak instrument
- **The window covers 40 % of a clip.** On median 23.8 s clips, ±5 s covers ~42 %. Random-time is 0.387–0.424, and
  the permutation hit p95 is 0.6 (stage 4) or 0.8 (stage 5).
- **The binomial floor** (p₀ = 0.4, `power.py`):
  - P(≥ 4/5) = 0.087;
  - P(5/5) = 0.010.

  So only 5/5 is evidence, and one miss is fatal.
- **A continuous rank discards less information.** Use the within-clip percentile of the label-window score, with
  a circular-shift null per clip, pooled over the 5 clips. It is still bounded by about 2–3 independent positions
  per 24 s clip: P(all 5 top by chance) ≈ 0.004–0.03.
- **Is a clip-level verdict the right target?** It probably is what the jury scores (clip yes/no plus a time), but:
  - on train-17, clip discrimination is confounded by scene and length;
  - event-level ±5 s adds little beyond the clip verdict at this clip length.

  Recommendation: evaluate the two **separately**.
  - A within-clip localisation test on the positives answers "does the score find the moment".
  - A length-calibrated clip test answers "does the score separate clips".
  - Do not merge them into one J = F1 + hit key over 24 configs.
- **Baseline.** A label-free position prior (t = duration − c, with c fitted in-fold) must be a localisation
  baseline (C5-4).

### (f) Power at n = 17 (5 / 12). COMPUTED (`power.py`, exact or analytic)
| Bar | What it requires |
|---|---|
| clip F1 > perm p95 0.667 (s5) / 0.727 (s4) | TP 3 + FP 0, TP 4 + FP ≤ 1 (≤ 2 for s5), or TP 5 + FP ≤ 3 (≤ 4 for s5) |
| clip F1 > duration-only 0.600 | TP 3 + FP ≤ 1, TP 4 + FP ≤ 4, or TP 5 + more |
| hit > perm p95 0.8 (s5) | 5/5 |
| hit > perm p95 0.6 (s4) | ≥ 4/5 |
| exact Mann–Whitney, single locked score, no selection | AUC ≥ 0.783 (α 0.05), AUC ≥ 0.867 (α 0.01) |
| sign test over 5 positives, one hypothesis | 5/5 → p 0.031; 4/5 → p 0.19 |
| positives needed for single-score AUC 0.75 / 0.80 / 0.90 (power 0.8, α 0.05, neg ≈ 2.4×) | ≈ 12 / 8 / 4 |

The nested F1 route with 24 configs and F1-max thresholds needs a cue that a single locked AUC test would detect at
p ≪ 0.01. Empirically (C5-2, S4-3):
- GO ≈ 93–100 % for a cue in 5/5 positives;
- 33 % for 4/5 (stage 4);
- 0 % for null.

A realistic, partial, behaviour-level cue (3/5) fails almost surely.

### (g) What the features carry, and what is unexploited
**Stage 4: YOLOX boxes, counts and motion.**
- What it extracts:
  - person boxes for 13 COCO classes (`train_probe.py:29`), sampled at ~4.3 fps (102 samples / 23.8 s);
  - a 48×27 motion grid, collapsed to 3 scalars: global, zone and ring mean (`:144-146`);
  - `objects_near` = the count of non-person COCO boxes in the ring (chosen in 14/17 folds).
- What it can carry about a cashier violation:
  - presence and absence (п.22, leaving the workplace);
  - a second person (п.42);
  - gross activity level.
- It cannot carry banknotes, recounting, drawer state or keys, because COCO has no such classes.
  `objects_near` most plausibly tracks counter clutter or scene identity, not an action.
- **Unexploited and cheap (existing caches, no decode):**
  - **The spatial motion grid.** Motion in the band between the operator zone and the customer side (a hand-over
    proxy), or zone-normalised motion maps. These are comparable across cameras.
  - **Person tracks.** Zone-person centroid displacement and box-height change (reaching or bending), and the length
    of `zone_empty` runs (it fires in 53 % of clips but only 6.5 % of samples).
  - **Detector confidence dynamics.**

  Each is one more hypothesis on the same 5 answers.

**Stage 5: SigLIP2 pooled 768-d, two views, PCA-8 or 8 prompts.**
- The pooled MAP-head embedding (`stage5_encoder.py:111,166`; ONNX output `emb`, `:176-178`) of a 224 px frame
  or a ×1.5 zone crop.
- What it carries: global scene semantics, for example "person at a desk" or "two people".
- At 224 px, a top-view cash counter puts banknotes and hands in a few patches, and pooling averages them away.
- **PCA-8 is unsupervised** and keeps the axes of largest variance, which are scene identity (a).
- **The 8 prompts are guesses**, and the text–image cosine of a pooled embedding is scene-dominated.
- **Unexploited:**
  - **Patch tokens.** They are not cached, because the ONNX export emits only the pooled output. Using them needs a
    new export (a new pinned artifact) plus re-embedding (~75 s per view for 17 clips, measured).
  - **Higher-resolution tiles of the counter band.** Several 224 crops per frame, about 4× embed cost, ~5 min.
  - **Temporal differences of patch tokens** (local semantic change).
  - **Norm-derived prompts.** Prompts tied to NORMS §3 behaviours (C-20/22/31-4/42), fixed *before* any run.
    These are still guesses, but not data-tuned.

## 2. Ranked directions
Ranked by expected gain | evidence | cost on existing caches | leakage/overfit risk. "Same answers" means the
direction is scored on the same 5 positive label times that stages 2/4/5 already used. It is then a re-test of the
same answers, at most a different hypothesis, and must be disclosed as such.

| # | Direction | Gain | Evidence | Cost | Risk | Same answers? |
|---|---|---|---|---|---|---|
| D1 | **Within-clip localisation test** on the 5 positives: label window vs same-clip rest; ONE pre-registered score, either (i) a label-free novelty score (‖e_t − median_clip‖, already computed as `dist-to-median`) or (ii) a fold-learned direction with clip fixed effects; continuous rank metric with a circular-shift null; position-prior baseline | low–moderate (answers "is the moment findable"; max p ≈ 0.01–0.03) | synthetic: scene-immune, rank 0.62–0.81 (claim b) | minutes, existing SigLIP + stage-4 caches | low leakage (no clip-level selection); overfit if more than 1–2 scores | **YES**: new hypothesis (within-clip), same 5 times; (i) is label-free, so it is the least post-hoc |
| D2 | **Metric/decision redesign**: single locked score, exact Mann–Whitney or permutation AUC primary; threshold at a fixed FP budget (e.g. ≤ 1/12) instead of F1-max; nested F1 secondary | moderate *in power* (bar AUC 0.78 instead of ≈ 0.93), zero *in signal* | `power.py`; C5-3 (TN = 0 collapse) | none (procedure only) | the choice of which score is post-hoc if it is taken from stage-4/5 winners (`objects_near`, `zone|E`) → forbidden | YES if applied to old winners; acceptable only with a score fixed a priori (D1-i, D3) |
| D3 | **Length-calibrated clip score** (per-clip circular-shift p-value of the max) for any clip-level verdict | removes the confound; does not create signal | claim (d); synthetic r(score, dur) 0.26–0.43 for the max | seconds | low | YES (clip labels) |
| D4 | **Unexploited cached structure**: spatial motion band, person tracks, zone-empty runs (stage-4 caches); norm-tied prompts (SigLIP caches, text tower only) | low–moderate; only if a hypothesis is fixed a priori from NORMS (п.22 absence, п.42 second person, п.20/31 hand-over) | none on real data; п.42 / second person already gave recall 0 on holdout in stage 1 | minutes | multiplicity: each feature is another look at the same 5 answers | YES |
| D5 | **Patch-token / high-res SigLIP2** (new export + re-embed) | unknown; addresses small objects | none; V1 4/5 unseen caps it | ~10–20 min incl. export pin and licence unchanged | new artifact to pin; same answers | YES |
| D6 | **More positives**: re-designate the frozen holdout (16 positives) into CV | **largest**: about 21 positives; single-score AUC 0.75 becomes detectable | `power.py` (≈ 12 positives needed) | cheap compute | **destroys the only quasi-independent check**; rule-blocked; needs explicit USER consent; was already seen once in stage 1 | NO (new answers), but spends the holdout |

**Not recommended (re-tests the same hypothesis on the same answers):**
- re-running stage 4/5 with other λ, W, PCA dimensionality or prompts;
- choosing `objects_near` or `zone|E|lam1` as "the" locked score;
- relaxing the target span or the J key;
- dropping the 2 zone-mismatch clips (that uses human R1, so it is not deployable);
- any VLM repeat (stage 3 NO-GO: false "yes" 56–81 % on absent controls).

### Falsifiable pilots (synthetic first, then train-17 under a locked protocol)
**P1 (D1 + D2 + D3, recommended single attempt).**
- **Synthetic validity** (`tools/make_synthetic.py` planted fixture, LOCAL_PRIVATE_DIR in scratch). It must show:
  1. power: planted cue → within-clip rank p < 0.05 in ≥ 80 % of replicates;
  2. null calibration: rank p uniform, false-positive rate ≤ 5 % over ≥ 200 null fixtures;
  3. scene control: a fixture with strong per-clip background differences and no cue → no signal;
  4. length control: the calibrated clip score has |ρ| < 0.2 with duration on null;
  5. timing: < 2 min.
- **Train-17 protocol** (locked, sha printed):
  - Hypothesis H1 (localisation): score (i), label-free `dist-to-median` on the zone view, one score with no
    learning. The label window's mean percentile among the clip's own valid steps, pooled over 5 positives, beats
    the circular-shift null at p < 0.05.
  - The H1 bar must also exceed the position-prior baseline. Use its pooled percentile, or require a better hit
    count than "t = d − c" with c fitted leave-one-positive-out.
  - H2 (clip, secondary, exploratory): the length-calibrated max of the same score gives AUC ≥ 0.783 (exact MW).
  - **No other hypothesis, no grid.** One run.
- **Falsified if** H1 p ≥ 0.05 or H1 ≤ the position prior.
- **Even H1 passing licenses only "a label-free novelty score ranks the labelled moment above the rest of the clip
  on 5/5 train positives".** It is not a detector GO, and it is not a clip-level claim.

**P2 (D4, only if P1 is done and the time budget allows; counts as a second attempt and must be disclosed).** One
norm-tied hypothesis fixed before any run, for example п.22 "the zone person is absent for ≥ X s while a person is in
the ring". The X value comes from the norm rationale, not from data.
- Synthetic: planted absence detected; null false-positive rate ≤ 5 %.
- Train: the same within-clip rank test.
- Bonferroni ×2 with P1.

**P3 (D6): user decision only.** If the user agrees to spend the holdout, run one locked single-score evaluation
(P1's H1/H2, frozen) on the holdout. That is a **confirmation** of P1, not a re-split for training; nothing is
tuned on it. This keeps the holdout's role: one look, frozen procedure.

## 3. Honest prior and the minimal credible deliverable
**Prior.**
- Two families (counts/motion; frozen global semantics) failed by wide margins: both observed values were below the
  null medians.
- V1 says the labelled behaviour was not visible at the label second in 4/5 positives.
- With 5 positives, the probability that any train-17 analysis before 07:00 UTC yields a detector that clears the
  existing clip-level bars is **≈ 5 % or less**.
- The probability that P1-H1 (within-clip localisation by a label-free novelty score) reaches p < 0.05 is
  **≈ 10–25 %**. That guess reflects that "something changes near a labelled moment" is a weaker and more generic
  claim. Even then, it is a 5-clip, same-answers, exploratory result.

**Minimal credible deliverable if nothing passes:**
1. The fail-closed product as is (`failed/no_validated_rule`), plus the stage 2–5 measured NO-GOs with their
   independent reviews.
2. This diagnosis as a quantitative "why":
   - the power table;
   - the scene-identity mechanism;
   - the duration artefact;
   - the evidence that representation is not the bottleneck (oracle vs learned).
3. A precise data requirement: about 8 positives for AUC 0.80, about 12 for 0.75, with a single locked score.
   Alternatively, user consent to spend the holdout once on a frozen P1 procedure.
4. Optionally, a clearly marked **review-assist mode**: rank moments within a clip by the label-free novelty score
   for a human reviewer. It is never a verdict, never `no_violation`, and carries the UNVALIDATED banner (as the
   demo-only path already does). It must not be enabled in production.

## 4. Requirements for any stage-7 protocol (for the coordinator and detection-architect)
- Pre-register at most 2 hypotheses, with Bonferroni across them.
- Separate the localisation test (within-clip, positives only) from the clip test (length-calibrated).
- Add the label-free **position-prior** localisation baseline (C5-4) and keep duration-only.
- Primary statistic: rank/AUC or within-clip percentile with an exact or permutation null. F1 at an F1-max
  threshold is reported, never the gate. Its collapse to always-positive under no signal (C5-3) makes it a
  structural handicap.
- Every label-free quantity used is computed on all clips before any label is read. This includes the ICC check in
  (a), novelty scores and calibrated maxima.
- Disclose "same 5 answers, 3rd/4th look" in every output, and the holdout-seen-once note (C5-9).

## 5. Evidence (commands and aggregate results)
- `python3 diag7/power.py`: the F1 combination table, MW critical AUC (0.783 at α 0.05 / 0.867 at α 0.01),
  binomial hit floors, and the n_pos-for-AUC table (§1f).
- `python3 diag7/scene.py 0` (30 reps, cue a ≤ 1.5): clip AUC ≈ 0.38–0.52 for all formulations. This is weak-cue
  noise.
- `python3 diag7/scene2.py` (20 reps):

  | Formulation | Metric | a = 3 | a = 5 |
  |---|---|---|---|
  | A (stage-5-like), σ_s 0 | AUC / hit | 0.77 / 0.77 | 0.93 / 0.96 |
  | A (stage-5-like), σ_s 3 | AUC / hit | 0.58 / 0.67 | 0.59 / 0.88 |
  | C (within-clip) | AUC | 0.54–0.57 | 0.65 |
  | C (within-clip) | rank | 0.69–0.70 | 0.78–0.81 |
  | C (within-clip) | r(score, dur) | 0.36–0.38 | 0.26–0.43 |
  | O (known direction, no learning) | AUC / hit | 0.92–0.93 / 0.99–1.00 | 1.00 / 1.00 |

- Real aggregates were quoted only from the two summary JSONs (fields named in §1).
