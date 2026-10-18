# PROTOCOL STAGE 9: exploratory small head on frozen embeddings (pre-registration)

**Status: DRAFT — NOT LOCKED. No real run without coordinator GO after an independent review.**
- Implementation: `tools/stage9_head.py`. Synthetic self-test: `tools/stage9_selftest.py`.
- **Number of attempts: 1.** Any change after a lock is a dated amendment appended at the end.

## 0. Feasibility verdict (stated before any real run)
**NOT TRAINABLE on the current snapshot. Expected outcome: `STOP_NOT_TRAINABLE`.**

Stage 8 closed with a measured `STOP_INSUFFICIENT_DATA` (lock 5c94785af7be6620). The coordinator recorded these
aggregates:
- 2 positives with numeric intervals;
- 0 confirmed-confident control clips, because the single same-clip control lies in an unknown-flagged positive
  task;
- 0 external controls.

The concrete reasons:
1. **No negative supervision.** The only allowed negatives are confirmed-confident controls from a clip with a numeric
   event interval (§2). There are none.
   - Unknown tasks are excluded entirely, together with their controls. Imported negatives are not negatives.
   - So every leave-one-out fold has 0 negative rows, and so does the resubstitution fit. A discriminative head is
     undefined.
   - This holds whatever mask is used. The mask is never relaxed to get around it.
2. **Only 2 positive intervals.** One is partial and both have an uncertain side. The strict inward margin removes
   more rows.
3. **No transferable test.** A fold holds out at most one positive, so no test, interval estimate, threshold or
   comparison with the prior is meaningful.

**A positive-only or resubstitution check is not proposed.** With no negatives, a fit on 2 intervals reduces to "the
direction of the interval frames against clip background". That is the same object as the stage-7 within-clip
novelty contrast, scored on the same clips. Stage 8 §6 already excluded it as uninformative at n = 2: a hit cannot
separate a violation from any scene change, and a miss does not show absence of signal.

This protocol and its code are kept as a record. They are a ready, tested machinery for a future snapshot that has
negatives. Running them on the current snapshot can only print `STOP_NOT_TRAINABLE`.

## 1. Scope and disclosure (binding)
- **Exploratory only.** The train set (C02 train, 17 clips) has been reused 5+ times: stages 4, 5 and 7, the human
  audit, and 8b. Nothing here is inference, and nothing may be presented as quality or as 5 confirmed events.
- **Holdout: CLOSED.**
  - Units come from `local_safe.select_split_units(side "train", dry_run=True)`.
  - Non-train snapshot records are discarded and counted.
  - Holdout videos, caches and scores are never opened or computed.
- **Production stays fail-closed** (`failed/no_validated_rule`). Verdicts stay strictly `violation | no_violation`.
  Nothing here enables a rule. Stage-5/7 GO bars are unchanged.
- **Frozen features only.** No new labels, no new encoder pass, nothing published.
- **Precondition:** `<LP>/train_probe/stage8_summary.json` exists and has a stage-8 outcome. Otherwise the run is
  refused.

## 2. Supervision mask (fixed before any result)
Labels come only from the pinned read-only snapshot. Its sha256 `31c0a7548a928fd2…` is verified before parsing, and
the latest revision per task is used. Both rules are shared with `stage8_audit`.

**Positive rows.**
- Source: done positive tasks of class `intervals` that are train `reviewed_violation` units with at least one
  interval whose boundaries are both numeric.
- A row is a frame whose time lies inside a numeric interval after shrinking it by **MARGIN = 5 s** on every side
  flagged `*_uncertain`.
- Weight: 1.0 if `visibility = good` and `confidence = sure`, otherwise 0.5.
- Intervals with an unknown boundary are excluded and counted.

**Negative rows.**
- A row is a frame inside a confirmed-confident same-clip control of such a task, using the stage-8 rule: both
  confirmations, good + sure, numeric boundaries, no overlap with any event interval.
- The frame must also lie inside no event interval of that clip. An event with an unknown boundary counts as open on
  that side.
- Weight: 1.0.

**Never rows of the loss:**
- unknown tasks and their controls (excluded entirely; they are not negatives);
- imported negatives;
- external control tasks (counted, not used);
- low-confidence, unconfirmed or overlapping controls;
- every other frame of a positive clip (unlabeled).

A frame that is both positive and negative is excluded and counted.

**Class balance:** weights are rescaled so that both classes have equal total weight.

**Strictness (coordinator constraint).** The temporal support of every row is the single frame it comes from (§3),
so a row never carries context from outside the confirmed interval.
- If no rows survive, that is the stop reason. The mask is not relaxed.
- Surviving row counts are reported per fold, as counts only.

## 3. Features and model (fixed)
**Features: point level.**
- One row per decoded frame from the stage-5 cache `cache_emb_siglip2/<sha16>.npz`: 1 fps, `t` = that frame's own
  decode time.
- Each row is L2-normalised `E_zone` minus the clip's own per-dimension median. This is label-free and uses that
  clip only.
- No temporal window or score smoothing is used. The stage-7 trailing 5 s mean is deliberately **not** used.

**Cache checks.**
- A cache is treated as missing (a failed unit, counted, never a negative) when it:
  - is absent;
  - is larger than 64 MB;
  - has non-finite values or non-increasing `t`;
  - has fewer than 3 frames.
- Only the caches of labelled positives and train imported negatives are read. Unknown-task clips are not loaded.

**Scaler.** Per-dimension median and robust sd: max(1.4826·MAD, 0.1·SD, 1e-6). It is fitted over all frames of the
fold's training clips only.

**Model.**
- L2-regularised logistic regression with a fixed **λ = 10** (one value, no grid) and an unpenalised intercept.
- Fitted by IRLS (Newton steps with step halving). The solve is reduced exactly to the row space of X.

**Score and localisation.**
- The score is the linear logit per frame.
- Localisation is the **earliest argmax** within the clip.
- A hit means the argmax lies within an interval ± **TOL = 5 s**, using the unshrunk interval.

## 4. Split and reported quantities
**Folds.**
- Leave-one-labelled-positive-clip-out. Windows or frames of one clip are never split between train and test.
- A fold is **not trainable** when it has 0 positive rows (`no_positive_rows_under_strict_mask`) or 0 negative rows
  (`no_negative_supervision`).

**Per trainable fold:**
- head hit;
- position-prior hit, using the frozen stage-7 relative rule
  t̂ = median_{training positives}(tl/d)·d_k, with tl := the midpoint of the first numeric interval;
- random expectation, |∪(interval ± TOL) ∩ [0, d]| / d;
- within-clip percentile: the share of frames outside interval ± TOL that score below the mean score inside the
  interval.

**Aggregates reported:**
- counts of hits (k / n_trainable) for the head and the prior;
- fold pattern counts (head hit/miss × prior hit/miss, plus not trainable);
- the mean random expectation;
- the sorted percentiles;
- rows (positive, negative) per fold;
- IRLS convergence.

**Exploratory extras, labelled, no inference:**
- **Resubstitution** (`RESUBSTITUTION_ONLY_NOT_VALIDATION`): a fit on all labelled rows; reports hits k/n and the
  percentiles.
- **FP screen** ("imported negatives are clip-level, unverified normal; screen only"): uses the resubstitution model.
  Reports the clip-max score of each imported negative as min/median/max, the positives' interval max, and the count
  of negatives whose clip-max reaches the smallest positive interval max.

**Outcome:**
- `STOP_NOT_TRAINABLE` with the reasons, when no fold and no resubstitution model is trainable;
- otherwise `EXPLORATORY_ONLY_VALIDATION_UNAVAILABLE`.

Every summary also carries the fixed statement "validation unavailable: …" with the counts. Seeds are not needed
because the method is deterministic.

## 5. Machinery and fail-closed
- `--print-lock` reads no data. It prints the sha256-16 over this file, `stage9_head.py` and `stage9_selftest.py`,
  plus the dependencies `stage8_audit.py`, `local_safe.py` and `annotate_intervals.py`.
- `--run --expect-lock <digest>` is refused, with exit 2, a static message and no summary, when:
  - the digest differs;
  - the summary already exists (single attempt);
  - the snapshot sha differs;
  - the split, labels or map are malformed;
  - stage 8 is not closed;
  - the 5 min budget is exceeded.
- Before and after the run, these are hashed: the 8 immutables (`annotate_intervals.IMMUTABLE`), the snapshot and the
  stage-8 summary.
- Output: `<LP>/train_probe/stage9_summary.json` (O_EXCL, 0400) and stdout. Both carry aggregates only: no shas,
  paths, names, per-clip rows, interval or label times. Errors print only the exception class.
- Budget: about 17 small caches plus IRLS in at most 2·n_rows dimensions. Expected about 1 s, hard bound 5 min.

## 6. Synthetic self-test (`python -m tools.stage9_selftest DIR`)
The CLI fixtures use `stage7_selftest.build`:
- 17 train + 4 holdout dummy clips, with the holdout at mode 000;
- synthetic 1 fps embedding caches, with durations of 60–140 s;
- a holdout cache at mode 000;
- a snapshot built with `clean_payload`, containing a superseded revision and a holdout record.

| Check | Setup | Expected |
|---|---|---|
| `cli_planted` | 3 numeric positives (one start-uncertain, one partial), 1 same-clip control in positive 0, 2 unknown, 12 negatives, planted cue | 3 folds, 2 trainable, 1 `no_negative_supervision`, hits 2/2 |
| `cli_real_like` | 2 numeric positives; the only control is inside an unknown task; 3 unknown | `STOP_NOT_TRAINABLE`, 0 trainable |
| Every CLI run | — | rc 0; summary 0400; open() spy finds no clip file, holdout path/cache or unknown-task cache opened; holdout stays 000; no sha16, sentinel, path or planted time in the outputs; second run, wrong lock, wrong snapshot sha and missing stage-8 summary each give rc 2 and no summary |
| `rep_planted` (in memory, 20 reps) | planted cue | head hit rate ≥ 0.9 |
| `rep_null` (60 reps) | no cue | hit rate within 3 binomial SE of the random expectation |
| `mask_rules` | — | margin on the uncertain side; weights 1/0.5; unknown tasks and their controls excluded; overlapping and low-confidence controls excluded; control rows outside events |
| `leakage` | perturb the held-out clip's labels and features, and the imported negatives | fold model identical |
| `print_lock` | — | opens no private file |

Result: 7/7 PASS (about 3 s). Four in-memory mutations were each caught: a scaler leak by `leakage`, a removed margin and unknown-as-labelled by `mask_rules`, and a zero cue by `rep_planted`.

## 7. Record
- Draft written 2026-10-18. Self-test 7/7; pytest 172 passed. The protocol sha changes with this line, so the lock digest is printed by `--print-lock`.
- Not locked, and no real run.
