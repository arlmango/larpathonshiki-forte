# Stage11: one exploratory exact-frame fit

Frozen before any stage11 model score. Codex takes over after the user's explicit instruction; Claude background task stopped, no concurrent fit. No Claude API billing.

Reuse pinned SigLIP2 ONNX, stage5 zone/preprocess and cached features. Train only: two numeric-positive clips and twelve explicit provider weak-negative clips. Three unknown tasks excluded, never negative. Labels are not shifted, widened or reinterpreted. Holdout closed; old artifacts and labels immutable.

Feature sanity first: sequential local decoding, native PTS in fixed [0,1 second) prefix for every selected clip, independent of label. Require all14 units, finite768-dimensional features, increasing PTS, no retrieval errors, exact-positive coverage5 frames, no label frames outside prefix, cosine drift>=.999. No frames, embeddings or raw labels leave the Mac.

Candidate support for every clip is every prefix frame plus20 rounded-linspace original cache rows. No label-dependent candidate selection. Positive training rows are only prefix frames within recorded positive-width bounds. Negative training rows are all candidates, weak clip labels only. Equal total weight per clip, then equal class weight; weight sum equals training row count. Lambda10 logistic IRLS from stage9; no hyperparameter search. L2 and own cached-clip median, robust scaler fitted only on training clips.

One leave-positive-and-negative-pair-out evaluation (24 fits): rank positive vs negative within the same model; AUC is mean pair win/tie. Fold threshold=max training-negative clip score. Each clip's prediction is median held-out margin>0; aggregate confusion/F1. Report exact-binomial marginal recall/specificity intervals, not independent generalisation estimates. Exact localisation fold hits descriptive (24folds share two positives). Duration AUC baseline. Repeated train reuse and tiny positive count prevent quality/GO claims.

One final fit on all14 train clips saves weights, scaler, config, input and artifact manifests. No deployed threshold. Reload must reproduce logits exactly. All old immutable hashes and run inputs checked before/after. Files private, read-only; attempt marker prevents automatic rerun.300-second fit budget. No next experiment automatically. Production remains fail-closed.

Coverage succeeded before fitting: feature manifest785de8bcfd1c481c,14units,positive2+3frames,finite768D,zero decode errors,42 drift rows minimumcosine1.0. Every logistic fit must converge; verify unchanged inputs before saving weights.
