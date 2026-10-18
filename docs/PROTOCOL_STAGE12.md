# Stage12: temporal multiple-instance head, one exploratory attempt

Fixed before the real fit on 2026-10-18. User explicitly authorised a new fit on all 17 real train clips.
No holdout features or videos are loaded; historical stage1 holdout exposure remains disclosed.

## Data and distinction from stage5/7/11

All 17 train clip labels imported from the original XLSX: 5 vendor-positive and 12 vendor-negative.
Source file hashes are checked against the existing split selector; training uses existing local pinned
SigLIP2 zone caches. Source table, caches, source files, interval snapshot and code enter the private input manifest.
Two user-refined intervals provide five exact positive frame embeddings from the pinned stage11 prefix cache.
The other three positive clips remain positive bags; unknown timing is never a negative label.
Vendor negatives remain weak labels, not verified normal intervals.

Stage5 used timestamp-window positive rows and row logistic regression; stage7 used a timestamp-window
within-clip contrast direction. Stage11 trained only two exact-positive clips with twelve weak negatives.
Stage12 instead optimises clip-level multiple-instance loss on ALL 17 bags, with separate auxiliary exact-frame
supervision. No presumed event window is manufactured for the three unknown-time positives.

## One fixed model and objective

Frozen encoder; only the head is trained. Per-clip L2-normalised embeddings are centred by that clip's own median.
This is offline full-clip inference, not streaming. The PCA12 projection is an uncentered SVD of the clip-median
residuals using only training-fold inputs, with equal mass per clip (not conventional globally mean-centred PCA).
At each original cached timestamp, features concatenate projected residual, trailing 2-second mean and absolute
deviation from that mean (36 dimensions). Auxiliary exact frames use the same projection and causal cached context;
when no earlier cached frame exists, context equals the exact frame itself. All feature scaling is learned only
on training clips, weighted equally per clip; standard deviations have a fixed 1e-4 floor.

Bag score = log(mean(exp(frame logits))), numerically stable, temperature 1. Candidate timestamps are the SAME
original 1fps cache policy for every clip, regardless of label or availability of exact annotations.
The exact annotated frames are auxiliary training inputs only, never label-conditioned evaluation candidates.

Loss = class-balanced clip BCE (each class total 0.5) + 0.1/2 times squared coefficient norm
+ 0.01 times clip-balanced mean squared differences of successive logits
+ 0.25 times auxiliary BCE. Auxiliary loss has equal total positive and negative mass:
positive exact frames normalised within each annotated clip, then equally across such clips; negative frames
normalised within each vendor-negative clip, then equally across negative clips. Recorded confidence weights
are preserved within positive clips. No frame outside an exact interval in a positive clip gets a negative target.
If a synthetic fold has no exact-positive auxiliary input, auxiliary loss is absent for both classes.

One deterministic zero-initialised BFGS fit, Armijo backtracking, at most 500 iterations, max absolute gradient
below 1e-6 required; fixed coefficients, no hyperparameter search, no result-driven retry. Overall fit budget 600s.
Research-only threshold is fixed logit zero; no deployed threshold exists. This pooling reduces count dependence
relative to raw max, but does not eliminate duration/scene confounding or sparse-event dilution.

## Evaluation, baselines and restrictions

17 leave-one-clip-out folds; held-out arrays, labels and exact intervals never enter fitted PCA, scaler or loss.
Then one final fit on all 17 train clips. Persist final weights, fold weights, config, input manifest and summary
in a new stage12 directory with exclusive creation; stage11 and old archives remain unchanged.
Check exact inference equality after reload, all input hashes and eight existing immutables.

Report OOF confusion matrix/F1/AUC, balanced accuracy, convergence, vendor-endpoint localisation within 5s
for five positive clips, and exact-interval localisation for the two refined clips. The original 1fps candidate
support hits zero of the two short exact intervals: exact localisation is structurally unavailable on this grid;
auxiliary supervision does not change that. Endpoint accuracy is not event-type or legal-norm accuracy.
Baselines: always positive, always negative, duration-only LOO (sign and F1 threshold fitted on training clips),
random candidate endpoint hit rate, first-candidate endpoint hit rate, shorter-duration AUC and score-duration correlation.

Repeated reuse of train data makes every CV result exploratory. No new GO criterion replaces prior gates.
Permutation controls and independent evaluation are not performed in this attempt; retain NO-GO/exploratory-only
regardless of a baseline win. Production stays fail-closed, no normative/type label is invented, holdout remains
closed pending a separate frozen final-evaluation decision. Training-resubstitution metrics are not validation.

Synthetic fixtures test gradients, class balance, unknown-time positive participation, held-out isolation and
reload only. They are never part of real training. Only aggregate counts/metrics may reach stdout; source identities,
raw labels, timestamps, embeddings, scenes and frames remain local.
