# Stage14: one fixed temporal-change experiment

Explicit user request after stage13. All40 unique clips (21+/19-), adaptive exploratory evaluation only;
no independent holdout remains. Preserve stage13 and old artifacts.

No appearance coordinates enter the head. From cached full-view SigLIP sequences, subtract own clip median,
then compute four scalar channels: residual magnitude, adjacent change magnitude, second difference magnitude,
and distance to own trailing2-second mean. These are semantic temporal-change proxies, not measured optical flow.
Remove each channel's own clip median and divide by max(1.4826MAD, .1std,1e-6), clip to +/-5.
Offline within-clip background uses no labels. Across-clip mean/std fit only on train-fold, equal clip mass.
No filename, duration, time position, label-dependent candidate selection or appearance-vector inputs.

One fixed model: four coefficients and intercept, normalized logmeanexp MIL temperature1, class-balanced clip BCE,
L2=1 on coefficients, zero-init BFGS gradient tolerance1e-6, fixed logit threshold0. No auxiliary frame labels;
exact intervals are evaluation only. No model family search or OOF threshold tuning. Same stage13 five stratified
whole-unique-clip folds seed13. One final fit on all40. All learned normalization excludes held-out clips.

Compare macroF1, positive F1, precision, recall, AUC, endpoint +/-5s, exact localization and duration baseline on
identical folds. Stage13 scores reloaded for comparison, not retrained. Preserve all input/artifact hashes and
verify exact saved-model reload. Old 1fps exact-support limitation remains. Production stays fail-closed regardless
of adaptive CV changes; no type/norm inference. Only aggregate stdout; all private data stay local.
