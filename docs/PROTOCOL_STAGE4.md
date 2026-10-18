# PROTOCOL STAGE 4: preceding-window candidates on TRAIN caches (locked pre-registration)

Status: **LOCKED 2026-10-18, written before any stage-4 computation on real data; amended once before the run
(§3, pre-run amendment, from the user's review).** Implemented by
`tools/stage4_probe.py`. The tool prints the sha256 prefix of this file at run time, so that any edit after the
run is visible. The real train caches are evaluated **once**. If a number disappoints, nothing in the family,
the selection rule, the metrics or the decision rule is changed afterwards. Any later change gets a new stage
number and a new holdout-free justification.

## 0. Scope and what this is NOT
- **Data.** Only the existing TRAIN numeric caches `local_private/train_probe/cache/<sha16>_4.npz` (17 units,
  5 `reviewed_violation`, 12 `reviewed_no_violation`). These caches hold timestamps, a float16 motion grid and
  detector boxes. They hold no images.
- **Units.** Units are chosen only through `tools.local_safe.select_split_units(LP, "C02", "train")`. With all 17
  caches present, the dry-run form is used and no clip file is opened at all. If a cache were missing, only the
  missing TRAIN clip would be decoded, in the `train_probe` child (job sent over stdin, stderr to DEVNULL), after
  the side check that runs before the file is opened. The holdout is never opened.
- **No VLM, no new features.** The features come from `tools.train_probe.features()` unchanged, on the same
  estimated zone.
- **Outputs.** stdout and `local_private/train_probe/stage4_summary.json` carry aggregates only: no per-clip
  rows, shas, paths or times.
- **Localization target.** This measures **timestamp-endpoint localization**: whether the detector's
  self-chosen time falls within ±5 s of the imported label second. It is **not** an event onset, not an event
  interval and not a legal violation. No onset truth is invented.
- **No normative rule** is enabled or validated by this stage. Production stays fail-closed.

## 1. Candidate family (fixed, enumerated: 14 × 2 × 2 × 2 × 2 = 224 configurations)
| Dimension | Values |
|---|---|
| feature (14) | persons: `n_persons`, `n_in_zone`, `zone_empty`, `n_outside_zone`, `n_in_ring`; motion: `motion_global`, `motion_zone`, `motion_ring`; areas: `person_area_mean`, `person_area_max`, `zone_box_area`; distances: `min_person_dist`, `zone_to_other_dist`; objects: `objects_near` |
| window W (s) | 5, 15 |
| direction | high (v = x), low (v = −x) |
| aggregation over the window | mean, max (of v) |
| offset of the scan point (s) | 0, −2.5 (the window ends `offset` seconds after the scan point τ) |

Threshold: one scalar per configuration, fitted inside the training fold (§3).

The following features are excluded a priori and were not chosen by looking at results:
- `bag_near`, `phone_near`, `bottle_cup_near`, `book_near`, `electronics_near`: subsets of `objects_near`.
- `zone_box_aspect`, `zone_person_score`: detector-geometry and quality proxies.
- `chair_in_zone`: a static object.

The 30 s window is **excluded**. A 30 s window before the label fits inside the clip for only 1 of 5 positives,
so it cannot be evaluated fairly.

## 2. Scanning rule: the detector localizes by itself
For one clip with sample times t_k and one feature x:
1. The scan points τ are all sample times t_k.
2. The window for τ is every sample with time in (τ + offset − W, τ + offset].
3. A window is valid only if it holds at least max(2, ⌈0.5 · W · f⌉) samples, where f = samples / duration. This
   is the half-window rule at the clip edges.
4. stat(τ) = agg over the window of v, where v = +x for direction high and v = −x for direction low.
5. Clip score = max over valid τ of stat(τ).
6. Localization t_pred = the **earliest** τ that reaches that maximum (the argmax).

The label time, the human T1/V1/R1 answers and any trimming hint are **never** inputs to the scan. Labels only
supervise the training folds and score the outer folds. If a clip has no valid window, its score is −∞ and it has
no time.

## 3. Nested leave-one-clip-out (unit = sha16; 17 outer folds × 16 inner folds)
**Pre-run amendment, 2026-10-18 (before any real stage-4 run), at the user's review of this section.**
- The original text selected c\* by resubstitution: F1, hit rate and AUC on the same 16 outer-training clips that
  also fitted the threshold.
- That is not nested validation, because there was no inner held-out prediction. The original rule is
  **withdrawn and replaced** by the scheme below.
- No real numbers were produced under the withdrawn rule. A synthetic self-test had not been run either.

For each outer held-out clip k, using the 16 outer-train clips only:
1. **Inner clip-LOO.** For each configuration c and each inner held-out clip j (j among the 16):
   - fit θ_c on the other 15 clips by maximizing clip F1; the candidate thresholds are those 15 scores, and ties
     go to the higher threshold, as in `train_probe.fit_threshold`;
   - predict clip j: inner verdict = score_c(j) ≥ θ_c, and inner localization = t_pred_c(j).
2. **Aggregate the 16 inner held-out predictions per c:**
   - inner clip F1;
   - inner hit rate, the share of inner positives with |t_pred − t_label| ≤ 5 s regardless of verdict;
   - for the third key, AUC of the scores over the 16 clips. It has no fitted parameter, so it is not
     resubstitution.
3. **Select c\*** lexicographically by (inner F1, inner hit rate, AUC), then by the lowest configuration index.
4. **Refit** θ_{c\*} on all 16 outer-train clips. This is a single outer prediction for clip k:
   - verdict = `violation` if score_{c\*}(k) ≥ θ_{c\*}, else `no_violation`;
   - t_pred = t_pred_{c\*}(k).
5. The **duration-only baseline** (§5) and **every permutation replicate** use exactly this scheme. A replicate
   shuffles the outer labels first, then runs the full nested procedure.

Notes:
- Scores and scan localizations are label-free functions of each clip's own cache. Only θ, c\* (and, for the
  duration baseline, the sign) depend on labels.
- The verdict values are strictly violation / no_violation. A clip with a missing or failed cache or zone is
  **excluded and counted**, never scored as no_violation.
- The primary evaluation uses all 17 units. The 15-unit zone-agree run (§6) is sensitivity only.

## 4. Metrics (outer folds only)
- **Clip level:** TP, FP, FN, TN, precision, recall and F1 over the 17 outer verdicts.
- **Timestamp-endpoint localization hit rate:** the share of positive clips with min_l |t_pred − t_label,l| ≤ 5 s,
  regardless of the predicted verdict (this isolates localization).
- **Event level:** `tools/evaluate.py` `match_events` / `_prf`, ±5 s, through `train_probe.metrics`. Each
  predicted-violation clip emits one event at t_pred, and every label time counts toward FN.

## 5. Baselines and controls
- **Always-positive:** every clip is a violation. Clip P/R/F1; no localization.
- **Random-time localization (analytic):**
  - The expected hit rate of a time drawn uniformly over the clip is |∪_l [t_l − 5, t_l + 5] ∩ [0, d]| / d,
    averaged over the positives.
  - The unclipped approximation 10 s / d is reported next to it.
- **Duration-only classifier (nested):**
  - The score is ±duration. The sign and the threshold are chosen inside each training fold by (inner F1, AUC)
    with the nested scheme of §3.
  - Clip P/R/F1. It has no localization.
- **Permutation control: the full nested procedure (§3–4), 1000 permutations, seed 0.**
  - Each permutation draws a random permutation π of the clips.
  - Clip i receives label y_{π(i)}.
  - If that label is positive, clip i also receives the pseudo label times (t_{π(i),l} / d_{π(i)}) · d_i, i.e.
    the relative label positions carried over to its own duration.
  - Every selection is re-made inside each fold.
  - The resulting null distributions of the outer F1 and the outer hit rate are summarized as p50 / p95.
  - p_perm = (1 + #{null ≥ observed}) / (1 + n).
- **Duration confound:**
  - The point-biserial correlation and the AUC of duration against the label class.
  - For each configuration chosen in any outer fold: the Spearman ρ between its clip score and duration across
    the clips. The fold-weighted median |ρ| and the max |ρ| are reported. A max over scan points grows with clip
    length, so |ρ| ≥ 0.5 marks the choice as **duration-dependent**.
- **Coverage diagnostics:**
  - The share of label times ≤ duration (review S3).
  - How many positives have a full 5 / 15 / 30 s window before the label.
  - Firing rates of the 14 features (review S5).
  - Samples per clip, durations, failures.

## 6. Sensitivity ablation (not deployable)
- The same nested procedure and permutation control are run on the 15 units where the estimated-zone centre falls
  in the frame half that R1 annotated. The 2 zone-mismatch units are left out.
- The agreement rule:
  - Zone centre (cx, cy), from the same `estimate_zone` as the features, in normalized frame coordinates.
  - R1 `bottom` agrees if cy > 0.5, `top` if cy < 0.5, `left` if cx < 0.5, `right` if cx > 0.5. `unseen` counts as
    a mismatch.
- This is labelled **ABLATION**. It uses human R1 answers to pick units, so it is not a deployable detector, makes
  no "oracle zone" claim and does not feed the GO/NO-GO decision. Hard clips are not discarded from the primary
  analysis.
- The audit files are only read. Their sha256 is checked before and after the run, and only `unchanged` or
  `CHANGED` is printed.

## 7. Decision rule for ONE locked holdout evaluation
**GO** only if all four hold on the primary analysis (§3, all 17 units):
1. outer clip F1 > always-positive clip F1;
2. outer clip F1 > permutation p95 of the outer F1;
3. outer hit rate > the random-time expectation;
4. outer hit rate > permutation p95 of the outer hit rate.

Anything else is a measured **NO-GO**. Even a GO only licenses proposing one fixed holdout evaluation, of the
configuration-selection procedure with its parameters frozen. Independent critical review must still approve
before the holdout is opened.

## 8. Disclosure and prior expectation
- The human audit was done on this same train sample: T1 on labels 4 end / 1 middle / 0 start; V1 4 unseen /
  1 partial / 0 visible; R1 for 17 clips, zone agreement 15/17. It informed this design: preceding windows ending
  at the label second, the offset, and the zone ablation. The outer-fold numbers are therefore **not fully
  independent** of the design choices, even though every parameter is selected inside the training folds.
- **V1 = 4/5 "unseen"** means the annotator could not see the labelled behaviour in 4 of 5 positives. A cue
  computed from person boxes and motion is therefore expected to be weak. **The prior expectation is low (NO-GO
  likely).**
- With n = 5 positives:
  - a nested F1 can beat always-positive (0.4545 here) by chance;
  - one extra hit changes the hit rate by 0.2.
- The permutation control, not the raw comparison to the baseline, is what makes a GO credible.
