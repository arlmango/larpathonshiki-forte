# Stage16 — one final fixed distribution/persistence classifier experiment

Fixed before fit on 2026-10-18. No subsequent tuning or retry selected by results.
Hypothesis: clip-wide persistence/distribution of semantic temporal change generalizes better than
stage14 MIL pooling of individual peaks. Keep stage15 final and corresponding five fold localizers frozen.
This changes clip classification, not localizer supervision or the candidate support.

Representation: existing stage14 four label-free robust within-clip temporal change channels;
for each channel take linear-interpolated 90th percentile, fraction strictly greater than 1,
and lag-one Pearson correlation. Constant-pair correlation is zero (denominator <= 1e-12).
These 12 scalars contain no filename, duration, frame count, absolute position or ground truth.
No optical-flow claim. Per-clip background preprocessing is offline and label-free.
Global feature mean/population standard deviation (floor 1e-4) fit ONLY on training clips.

Classifier: linear logistic regression, intercept unpenalized, weighted BCE plus 0.5*sum(beta**2)
(L2=1). Each class has mass 0.5, equal clip weight within class. Zero initialization, existing BFGS,
max500 iterations, gradient infinity norm <1e-6. Fixed logit gate >=0. No grid, threshold tuning,
OOF selection or test-time labels. Six classifier fits: same five stratified whole-unique-clip folds
(seed13) as stage13/14/15, plus final fit on all40. Total fit budget 120 seconds, no new extraction/download.
Frozen localizers were supervised only by the corresponding train-fold vendor timestamps in stage15;
load the same fold's localizer, never final weights for held-out prediction.

All40 unique clips:21 positive/19 negative,22 distinct GT events. Evaluate unchanged 1fps candidates.
Emit at most one earliest argmax localizer candidate only when classifier gate opens. Match within clip,
nearest distinct GT within +/-5 seconds inclusive, ties earlier GT; one-to-one, unmatched GT=FN,
unmatched candidate=FP. Mistimed positive prediction produces FP and FN. All22 events in recall.
Report clip P/R/F1/macroF1/AUC/confusion, E2E event P/R/F1, all21-positive localization independent of gate,
all-positive ungated event metrics, gate misses and negative false alarms. Event AUC is not defined for
this single-candidate matching contract; AUC refers to clip logits only. Report exact interval diagnostic.
Compare saved stage15 on identical folds, duration-only clip baseline and always-positive clip baseline;
always-open gate plus frozen localizer is an additional event baseline. Baselines never drive selection.

Save new final/CV combined weights, protocol/config/input/artifact manifests, summary and exact reload.
Do not overwrite stage15 or previous archives. Keep negative result if inferior; no more experiments.
Adaptive reused CV, no clean holdout, NO-GO, production remains fail-closed. No automatic model promotion.
