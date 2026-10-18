# PROTOCOL STAGE 10: one exploratory weakly supervised fit (pre-registration)

**Status: DRAFT — NOT LOCKED. No fit has been run. A real run needs coordinator GO after the independent pre-lock
check.**
- Implementation: `tools/stage10_weak.py`. Synthetic self-test: `tools/stage10_selftest.py`.
- **Number of attempts: 1.** Any change after the lock is a dated amendment appended at the end.
- Basis: the independent reviewer's admissibility verdict (REVIEW_CRITICAL, Stage 10) and the user's pre-lock
  requirements. Both are folded in below.

## 1. Scope and disclosure (binding)
- **Exploratory only.** The C02 train set has been reused 5+ times: stages 4, 5 and 7, the human audit, and 8b. The
  design came after stage 7. Nothing here is inference, a detector, generalisation, a validated FP rate or recall,
  or significance.
- **Outcome labels:** `EXPLORATORY_TRANSFER_OBSERVED` or `EXPLORATORY_TRANSFER_NOT_OBSERVED`.
  - A non-evaluable run gives `STOP_NOT_EVALUABLE` or `FAILED_<status>`.
  - The run is never GO. The GO bars of stages 5 and 7 are untouched and not claimed.
- **Holdout: CLOSED.**
  - Units come from `local_safe.select_split_units(side "train", dry_run=True)`.
  - Holdout records are discarded and counted.
  - Holdout videos, caches and scores are never opened or computed.
- **Production stays fail-closed** (`failed/no_validated_rule`). Verdicts stay strictly `violation | no_violation`.
  Nothing here enables a rule.
- **Model weights are never written** (`model_artefact_saved: false`) and nothing is published.
- **Weak labels.** The 12 negatives are the dataset provider's explicit whole-clip "no violation" statements. They
  are **human-unconfirmed** and are never called "confirmed normal".
- **Correction kept.** The earlier stage-9 `NO_TRAINABLE_SUPERVISION` was too broad. It holds only for the paired
  interval contrast (0 confirmed controls). This is a different supervision type.

## 2. Inputs and pins (checked at run time; refused on mismatch)
| Input | sha256-16 |
|---|---|
| Intervals snapshot `labels/snapshots/C02.intervals.v1.20261018T0333Z.jsonl` | 31c0a7548a928fd2 (full sha pinned in stage8_audit) |
| Split `splits/C02.json` | 211d259b64882726 |
| Imported labels `labels/C02.imported.jsonl` | 329bee4969cdc19e |
| Manifest of the 14 used SigLIP2 zone caches (`cache_emb_siglip2/<sha16>.npz`; lines `sha16 sha256(file)`, sorted) | 2cf264ba8c91d029 |
| **Combined inputs digest** (`PIN10["C02"]`) | **735de57dbc2ee643** |

- **Recomputation.** `--run` recomputes the combined digest before the child starts. A mismatch is refused (exit 2, no
  summary). The digest is recomputed again after the run and reported as `unchanged_after_run`.
- **Also hashed before and after:** the 8 immutables (`annotate_intervals.IMMUTABLE`) and the stage-8 summary.
- **Precondition:** stage 8 is closed (`stage8_summary.json` has a stage-8 outcome).

## 3. Units and supervision (fixed)
**Units are clips, deduplicated by content hash.** The unit key is the sha16 of the file bytes. A byte-identical pair
is one unit, so it can never sit on both sides of a fold.

Pre-lock count (aggregates from `--prelock-duration-auc`):
- **2 positive units:** the done interval tasks with a numeric interval.
- **12 negative units:** 13 import rows, one of them a byte-identical duplicate.
- 0 caches missing.

**Positive instances.**
- A positive instance is a frame (1 fps, the frame's own decode time) inside a confirmed numeric interval.
- The interval is first shrunk by **MARGIN = 5 s** on every side flagged `*_uncertain`. This is the stage-8/9
  strict mask, and the frame is its own temporal support.
- Within-clip weight: 1.0 for good + sure, 0.5 otherwise. Each fit has exactly one training positive clip, so the
  0.5 weight has no effect. This is stated, not tuned.

**Weak negative instances.** All frames of each provider-explicit no-violation clip (bag label), weight 1.0 within the
clip.

**Never rows of the loss, never negatives, never FP:**
- the 3 provider-positive clips without a numeric interval (unknown / not_visible tasks). Their caches are not even
  opened;
- every frame of an interval clip outside its shrunk interval. These frames are unknown;
- same-clip and external controls (not used);
- holdout records.

Frames are never units.

## 4. Features, weights, model (fixed)
**Features.** The frozen stage-5 cache `E_zone`, at point level:
- each row is L2-normalised, minus the clip's own per-dimension median;
- this is label-free and uses that clip only;
- no temporal window (stage 9 `point_prep`).

**Scaler.** Per-dimension median and robust sd, as in stage 9: max(1.4826·MAD, 0.1·SD, 1e-6).
- It is fitted on the **training clips of the fit only**.
- Each training clip contributes its K grid frames (§5), so every clip counts equally.

**Weights.**
- Each clip's rows sum to 1, so all clip weights are equal.
- Classes are rebalanced to equal total weight.
- Weights are then scaled to sum to the number of rows, which keeps λ on the stage-9 scale.

**Model.**
- L2 logistic regression, **λ = 10** fixed (no grid), with an unpenalised intercept.
- Fitted by IRLS with step halving, reduced exactly to the row space (stage 9 `fit_logreg`).
- Score = the linear logit per frame.

## 5. Validation and reported quantities (fixed before any score)
**Clip score.** The max logit over **K = 20 evenly spaced frames**, at indices round(linspace(0, n−1, 20)). Short
clips repeat frames. Using a fixed K removes the max-over-length bias.

**Leave-pair-out (LPO).**
- For every (positive i, negative j) pair, there are 2 × 12 = 24 pairs. Each pair gets **one model**, trained on the
  other 12 units: 1 positive and 11 negatives.
- **LPO-AUC** is the share of pairs with s_i > s_j (ties count ½).
- Pooled leave-one-clip-out AUC is not used.

**Threshold rule.** θ = the max clip score of that pair model's training negatives. A held-out clip is flagged when
its score is above θ.

**Clip-level counts.**
- Each clip gets the median, over the pair models in which it was held out, of (score − θ). A clip is positive when
  that median is > 0.
- Reported: TP/FN/FP/TN, recall k/2, FP k/12 with a 95 % Clopper–Pearson CI (0/12 → [0, 0.2646]), and F1 vs
  always-positive F1 = 4/16 = **0.25**.
- Also reported: the pair-slot counts flagged.

**Duration-only baseline.** The AUC of clip duration ("longer = positive") on the same 14 units. It was computed
before the lock from durations only:

**AUC = 0.125, separability max(A, 1−A) = 0.875.**
- Positives are shorter, and duration alone orders 21 of 24 pairs.
- Separability is not 1.0, so a clip-level comparison is attainable. However, any LPO-AUC ≤ 0.875 says nothing
  beyond duration.
- The run reports `lpo_auc_minus_duration_separability`. It also reports Spearman ρ(median held-out clip score,
  duration), descriptive.

**Localisation, per held-out positive.**
- The model is leave-one-positive-out: trained on the other positive plus all 12 negatives.
- The top frame is the earliest argmax over all frames.
- `exact_interval_hit`: the top frame lies inside the confirmed numeric bounds.
- `tolerance_hit`: the top frame lies inside the expanded support, i.e. the interval widened by 5 s on an uncertain
  side and 2 s on a certain side.
- Each hit is compared with the uniform chance share of the **same** support: the exact share against the exact
  hit, and the expanded share against the tolerance hit. The all-positive products of both shares are reported.
- Frozen position prior: t̂ = (the other positive's first-interval relative midpoint) × duration. Its exact and
  tolerance hits are reported.
- Errors are distances to the confirmed interval (0 inside).
- In-interval vs out-of-support logit gap, plus the within-clip percentile (descriptive only; the outside frames are
  unknown, NOT negatives).
- (ii): for each positive, the share of its pair models in which its in-interval max beats the held-out negative's
  clip score (asymmetric: all in-interval frames vs the negative's K = 20 grid; favours positives; descriptive).
- Pair-model exact hits are counted from the same 24 models, with no extra fit.

**Outcome rule (pre-registered).** `EXPLORATORY_TRANSFER_OBSERVED` holds iff, **for every held-out positive**, all
four of these are true:
1. `exact_interval_hit`;
2. in-interval mean logit > out-of-support mean logit;
3. model localisation error < frozen position-prior error. If the prior also lands inside the interval, both errors
   are 0, so this fails;
4. expanded-support share of the clip < 0.5.

Otherwise the outcome is `EXPLORATORY_TRANSFER_NOT_OBSERVED`, with the failed conditions as sub-reasons.

**No p-values and no permutation.** Scene, duration and reuse break exchangeability. Under exchangeability alone, a
perfect clip AUC would have a nominal chance of 1/C(14,2) = 1/91. The brief's 200-permutation "scene-identity check"
is therefore dropped. The self-test shows why: positives sharing a scene inflate the clip AUC without any action cue.

**Reporting template (fixed).** A hit means only this: on this reused train set, a direction fitted on one confirmed
interval vs the provider-negative clips ranked another clip's confirmed interval above (i) that clip's own other
frames and/or (ii) the provider-negative clips.
- Without (i), it means scene similarity.
- A miss shows nothing.

**Open risk.** It is unverified whether the 2 positive clips share a camera or site distinct from the negatives. This
cannot be checked without viewing. Clip-level numbers (LPO-AUC, F1, FP) are therefore uninterpretable as action
evidence. The within-clip criteria (i) exclude only a static per-clip scene offset; they do not exclude time-varying
activity or novelty shared by the positive clips (self-test `scene`: false OBSERVED 0.10; cf. stage-7 N1), so even
OBSERVED does not establish violation specificity.

## 6. Machinery and fail-closed
- `--print-lock` reads no data. It prints the sha256-16 over this file, `stage10_weak.py` and `stage10_selftest.py`,
  plus the dependencies `stage9_head.py`, `stage8_audit.py`, `local_safe.py` and `annotate_intervals.py`.
- `--prelock-duration-auc` reads the snapshot, split, imported labels and the used caches. It prints only the unit
  counts, the duration AUC and separability, and the pin digests. It writes nothing.
- `--run --expect-lock <digest>` is refused, with exit 2, a static message and no summary, when:
  - the lock digest differs;
  - the summary already exists;
  - the snapshot sha differs;
  - the inputs digest differs;
  - stage 8 is not closed;
  - the split or labels are malformed.
- The computation runs in a child: the job goes via stdin, stderr goes to DEVNULL, and the child gets its own
  session. It is killed at **300 s**.
  - The internal deadline is 290 s.
  - On timeout or failure, a summary with `FAILED_<status>` and no metrics is still written (O_EXCL, 0400). This
    consumes the single attempt.
- Output: `<LP>/train_probe/stage10_summary.json` (O_EXCL, 0400) and stdout. Both carry aggregates only: no shas of
  units, no paths, names, per-clip rows, or interval, label or argmax times. The only per-positive values are sorted
  shares, gaps and percentiles.
- Budget: 26 fits (24 LPO + 2 leave-one-positive-out). About 3 s on the 768-d synthetic fixture.

## 7. Synthetic self-test (`python -m tools.stage10_selftest DIR`)
**CLI fixture.**
- Built by `stage9_selftest.fixture` on top of `stage7_selftest.build`: 17 train + 4 holdout dummy clips, with the
  holdout and one holdout cache at mode 000, a holdout record and a superseded revision.
- 2 interval positives (one start-uncertain, one partial), 3 unknown tasks, and 12 negative rows. Two of the rows are
  made byte-identical, giving **11 units**.
- 768-d caches with a planted cue.

In-memory checks use a 256-d generator with cue 0.25.

| Check | Expected | Result (2026-10-18) |
|---|---|---|
| `prelock` | rc 0; n = 2/11; 13 caches in the manifest; no file written; no leak | PASS |
| `open_spy` (in-process compute) | no clip file, holdout path/cache or unknown-task cache opened; holdout 000 | PASS |
| `cli_planted` | OBSERVED; 22/22 pairs; units 2/11 (dedup); unknown excluded 3; summary 0400; integrity and inputs verified/unchanged; no weights; no leak. Refusals each rc 2 with no summary: second run, wrong lock, wrong snapshot sha, stage-8 summary missing, wrong inputs digest, tampered used cache | PASS (runtime 2.8 s) |
| `rep_planted` (10) | OBSERVED ≥ 0.9; exact hits ≥ 0.9; LPO-AUC ≥ 0.9 | PASS: 1.00, 20/20, 1.000 |
| `rep_null` (40) | exact hit rate within 3 SE of chance; false OBSERVED ≤ uniform both-hit + 3 SE | PASS: 12/80 = 0.150 vs 0.179; false OBSERVED 0.050 vs 0.032 |
| `duration` (10) | separability 1.0 → descriptive flag; K-grid less length-biased than an all-frames max | PASS: \|AUC−0.5\| 0.283 vs 0.371 |
| `scene` (10) | LPO-AUC inflated (≥ 0.7) while the exact hit rate stays within 3 SE of chance | PASS: AUC 0.829; hits 0.350 vs 0.174; false OBSERVED 0.10; percentile 0.51 |
| `mask_weights` | margin; outside frames never rows; unknown not units; equal clip weights; classes balanced; Σw = n; K = 20; Clopper–Pearson known values for n = 12 at k = 0/1/6/12 | PASS |
| `leakage` | pair model identical when the held-out pair's labels and features change | PASS |
| `print_lock` | opens no private file | PASS |

**Disclosed limits.**
- In the synthetic scene and null fixtures, the false OBSERVED rate (0.05–0.10) is near the uniform both-hit
  reference (about 0.03), but not below it. With n = 2 held-out positives, OBSERVED remains weak evidence.
- The K = 20 grid only reduces the length bias of the clip score; it does not remove it (synthetic duration
  fixture, median ρ 0.21).
- The grid can skip a short interval in a long clip. This affects the clip score only, not localisation.

## 8. Record
- Draft written 2026-10-18 with the reviewer's requirements and the user's pre-lock fixes folded in. Those fixes
  are: the Clopper–Pearson correction, no redundant refits, exact vs tolerance hits, the OBSERVED rule, and the
  input pins.
- The pre-lock duration AUC and the pins were computed locally. Only the aggregates above were seen.
- Not locked, and no real fit was run. The protocol sha changes with this line, so the lock digest is printed by
  `--print-lock`.
