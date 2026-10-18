# Stage13 all40: explicitly authorised adaptive final training

2026-10-18: user explicitly authorises all 40 unique real ForteBank clips, including the 23 former holdout clips.
The original split remains unchanged on disk, but is no longer an independent evaluation set. All metrics are
exploratory/adaptive. The 42 physical files are not treated as 42 independent clips; content-SHA selection yields
40 units: original train17 (5 positive/12 negative) plus former holdout23 (16 positive/7 negative), total21/19.

Data: existing full-view SigLIP caches for17, missing23 extracted locally with the SAME pinned encoder,
preprocessor and 1fps sample rule from stage5. No downloads. Full-frame view is uniform across all40 to avoid
mixing old zone-view with newly missing zone estimators. This is the sole representation-view change from stage12,
chosen before results for consistent coverage, not selected by CV. Two exact-positive auxiliary clips are also
re-extracted in full-frame view and centred by their corresponding full-view cache median. All unknown-interval
positive clips remain positive MIL bags. No negative targets are invented outside intervals in positive clips.

Head, optimiser, loss weights and threshold are exactly stage12: uncentered residual SVD12, current+trailing2s mean
+absolute deviation, training-fold equal-clip scaling, logmeanexp temperature1, clip-balanced BCE, L2 .1,
auxiliary .25, smoothness .01, zero-init BFGS with gradient tolerance1e-6, research threshold0. No tuning or retries.
Five stratified folds, fixed seed13, whole unique clips only. All PCA/scaler/head learning excludes held-out clips.
One final fit on all40. Baselines always-positive/negative and duration-only using the SAME five folds; duration
sign/threshold learned only on training clips. Report confusion/AUC/F1, endpoint and exact localization, runtime,
coverage and exact saved-model reload for final and all CV folds. Candidate timestamps stay label independent.
The old1fps limitation for two short exact intervals remains disclosed; auxiliary frames do not enter candidates.

New exclusive-create artifacts only: research_models/stage13_all40 and train_probe/stage13_all40. Preserve original
split, labels, stage11/12 weights/results and all archives. No production GO: independent validation and previous
quality gates remain unsatisfied. No event type or legal norm inferred. Raw video/frames/scenes/labels/embeddings
remain local; stdout contains aggregates only. Missing extraction/timeout is an explicit blocker, never a negative.
