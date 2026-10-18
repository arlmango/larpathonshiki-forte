# Stage15: frozen clip classifier plus separately supervised temporal localizer

Fixed before fit, 2026-10-18. One method, one real attempt. All40 unique real clips are authorised; the former
holdout was already consumed in stage13. Every reported CV metric is adaptive/exploratory, not an independent test.
Preserve stage12/13/14 artifacts and all source archives. Production remains fail-closed and NO-GO.

## Frozen classifier and localizer

Classifier: copy the saved stage14 final and five fold models without fitting or changing them. Fixed gate is
clip logit >= 0. Use the SAME stratified whole-content-clip folds from stage13/14, seed13. Each held-out clip is
scored by its corresponding classifier fold, never the all40 classifier. This pairing is selected before results;
there is no post-hoc selection between old OOF localizations.

Localizer: eight scalar features, the four stage14 change/background channels plus their causal trailing3-second
means. No appearance coordinates, filename, clip duration, absolute/relative position or ground truth at inference.
Within-clip background normalization is label-free and offline. Global equal-clip mean/std is learned on each
training fold only, std floor1e-4. The same label-independent cached1fps candidate support is used for every clip.

Supervision: vendor timestamps of TRAINING-fold positive clips only. Positive rows lie within +/-2s inclusive of
any vendor timestamp. Other positive-clip rows are unlabeled and excluded from supervised loss. All rows of vendor-
negative training clips are weak negatives. Weight each positive clip equally within total positive mass0.5 and
each negative clip equally within negative mass0.5; then equally weight supervised rows within that clip. Multiple
vendor windows are unioned, not duplicated. No confidence or norm/type is inferred. Exact human intervals are used
only for an extra evaluation diagnostic, not to shift vendor endpoints or make inference candidates.

One linear localizer, eight coefficients and intercept, average weighted BCE plus L2=1/2 times squared coefficient
norm, intercept unpenalized. Deterministic zero-init BFGS, gradient infinity norm<1e-6, max500 iterations. No parameter
search, threshold search or retry chosen by results. Five fold localizer fits plus one final all40 fit.

Coverage before fit: all21 positive clips have timestamps,22 distinct events total (one clip has two); all21 have
cached candidates within +/-2s (88 positive rows);19 vendor-negative clips have no event timestamps. If this changes,
refuse rather than silently exclude units or widen training windows.

## Fixed emission, matching and metrics

Localizer always computes one earliest-argmax candidate. End-to-end emits that candidate only if the frozen
classifier gate is open. No localizer threshold or top-k search. Thus one clip can emit at most one event; the
second GT event of a multi-event clip remains a possible FN and cannot be counted twice.

Matching: within each clip only, tolerance +/-5s inclusive. With at most one prediction, match the nearest distinct
GT timestamp within tolerance; ties go to the earlier timestamp. Each prediction and GT can match once only.
Unmatched predictions are FP, unmatched GT events FN. A mistimed event in a positive clip counts both FP and FN.
Negative-clip emissions are FP. Aggregate micro event precision/recall/F1 over ALL40, with all22 GT events in recall.

Also report: unchanged clip confusion/positiveF1/macroF1/precision/recall/AUC; localization hit rate and nearest-error
summary across ALL21 positives independent of gate; ungated-positive event metrics; exact-interval hits (2 clips,
known zero1fps exact support); positive gate misses (clips and GT events), localization misses among gate-open
positives, and false alarms among19 negatives. No conditional localization statistic substitutes for all21 results.

Fixed descriptive baselines: frozen stage14 classifier plus its original argmax on the same folds, and duration-only
clip classifier fitted inside each train fold (not used by detector). Compare both with consistent event matching.
The one-prediction limit and weak timestamps restrict interpretation. No GO criterion is lowered.

## Artifacts and research adapter

New research_models/stage15_two_component contains combined final/fold classifier+localizer weights, config,
input/artifact manifests and aggregate summary. Preserve stage14 classifier arrays exactly. Check final and CV
reload equality, input hashes, previous artifacts, and deterministic scaler/gradient invariants.

Separate tools.stage15_api factory accepts only opaque IDs resolved by an injected local provider. No video,
embeddings, paths or labels accepted in HTTP payload. Every successful response is research_only=true,
production_enabled=false,status=not_validated, with uncalibrated clip logit and at most one unvalidated candidate.
No violation/no_violation verdict, invented type or norm. Missing/malformed provider fails closed with static errors.
Not mounted in app.main, not started or deployed. Synthetic tests only exercise the adapter.
