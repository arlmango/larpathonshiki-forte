# PROTOCOL STAGE 5: frozen SigLIP2 image semantics + temporal features + small regularised model (pre-registration)

Status: **LOCKED 2026-10-17 20:29 UTC (2026-10-18 01:29 Astana), before any stage-5 computation on real data.**
Before the lock, the only computations were the synthetic smoke (§11) and synthetic self-test debugging runs.
`tools/stage5_probe.py` prints this file's sha256-16 at run time. After the lock, any change is a **dated
amendment appended at the end of this file**, never a silent edit. After the real run, nothing in §1–§9 changes;
a different family or selection rule needs a new stage.

**Pre-lock amendments, all applied before the lock and before any real computation** (coordinator, user reviews):
- (a) Zone policy is unambiguous (§0, §1).
- (b) Fail-closed decoding and the incomplete-primary policy (§0, §7).
- (c) One global 90-min wall-clock budget; n_perm is fixed from measured synthetic timing (§6, §11).
- (d) Bounded power (§9).
- (e) Audit wording, which only constrains interpretation (§7, §12).
- (f) The holdout stays closed even under an exploratory GO (§0, §7).
- (g) The ONNX artifact is pinned by full sha256 and never re-exported over (§1).
- (h) Precise duration-ρ definition (§6).
- (i) Planted self-test without the cue in negatives; a corrupt-clip scenario was added (§9).

**Disclosure (binding, repeated in every output).** This is **adaptive exploratory research after the stage-4 NO-GO,
on the same train sample** (C02 train, n = 17, 5 positives). The human audit (T1 4 end / 1 middle; V1 4 unseen /
1 partial; R1 17) and the stage-4 result informed the design. No independent validation is claimed. The holdout
stays unopened (it was already seen once in stage 1, so it is not fully virgin). A GO here only licenses proposing
one locked holdout evaluation after independent review. **Even under an exploratory GO, the holdout stays closed until
a separate agreement.** No normative rule is enabled. Production stays fail-closed.

Two further disclosures:
- **Inner-selection optimism exists.** Inside a training fold, the models that score inner clip j's neighbours were
  trained with j's label (§5.2). This inflates the inner selection key only. The outer clip is never involved.
- **The 8 text prompts are guesses.** They are not derived from any viewing of real frames, labels or scene
  descriptions; nobody on the design side has seen the data.

## 0. Scope and boundary
- **Units.** Units come only from `tools.local_safe.select_split_units(LP, "C02", "train")`. The side and index are
  checked before any file is opened, and the sha is re-checked after opening. Holdout records are discarded at parse
  time. Units: 17 (5 `reviewed_violation`, 12 `reviewed_no_violation`).
- **Embedding pass (one per clip, label-free).** It runs in a child process: the job goes over stdin, stderr goes to
  DEVNULL, and stdout is numeric only.
  - Frames are sampled at t_k = k + 0.5 s while t_k < duration, decoded sequentially with grab/retrieve.
  - Each frame is embedded in two views (§1).
  - The cache `local_private/train_probe/cache_emb_siglip2/<sha16>.npz` holds t, E_full and E_zone (float16, 768-d)
    and nothing else: no image, no label.
- **Outputs.** stdout and `local_private/train_probe/stage5_summary.json` carry aggregates only. They contain no shas,
  paths, names, per-clip rows, per-clip times, frames or embeddings.
- **Immutable files.** These are read-only: `labels/*`, `inventory_map.json` and `train_probe/stage4_summary.json`.
  Their sha256 is checked before and after, and only `unchanged` or `CHANGED` is printed.
- **Localisation target.** The target is the **timestamp endpoint**: the self-chosen time within ±5 s of the
  imported label second. It is not an event onset, interval or legal violation, and no onset truth is invented.
- **Verdict values.** Verdicts are strictly `violation` / `no_violation`. A missing or invalid model is a failure,
  never no_violation.
- **Zone failure policy (single, unambiguous).** If the estimated zone is missing (no stage-4 cache, or
  `estimate_zone` returns None), that clip's zone view **falls back to the full frame**. It is counted as
  `zone_fallback` and is **never excluded**. Hard clips are not discarded.
- **Decode / embedding failure policy (fail-closed).**
  - The child checks completeness. The expected samples are ⌈d − 0.5⌉ for d = the probe/inventory duration from the
    stage-4 cache (else the container frame count / fps).
  - A unit is `decode_incomplete` if any of these hold: a retrieve fails; decoded samples < expected − 1; or the
    decoded duration differs from d by more than 1 s.
  - For a `decode_incomplete` unit, no usable cache is written. The unit is counted, and it **remains a failure of the
    primary set**: it is not silently dropped.
  - Any embedding, timeout or memory-cap failure is treated the same way.
  - **GO is not issued if any of the 17 primary units failed.** Metrics on the completed subset are then reported
    only as "INCOMPLETE PRIMARY".

## 1. Encoder (frozen, no fine-tuning)
- **Model.** `google/siglip2-base-patch16-224` @ `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. It is loaded natively by
  transformers 5.20.1 as `SiglipModel` (`model_type: siglip`, no `auto_map`), with `local_files_only=True` and
  without `trust_remote_code`.
- **Inference.** The vision tower runs as ONNX (exported once from the pinned weights, torchscript exporter, opset 18,
  `weights/siglip2_base_patch16_224/onnx/vision.onnx`, sha256
  `4e95b080c23eff44205bd7fc87cc7818f1d613c559d8380de204db82374ffdba`; the embedding child refuses on mismatch and the
  smoke never re-exports over it) on onnxruntime 1.31.0
  `CPUExecutionProvider`. Parity with torch on 48 synthetic frames: cosine min 1.000.
- **Embedding.** The pooled image embedding (768-d) is L2-normalised.
- **Preprocessing** (`stage5_encoder.preprocess`):
  - BGR→RGB, then cv2 `INTER_AREA` resize to 224×224, then x/127.5 − 1.
  - Cosine against the HF PIL processor on synthetic frames: min 0.992. This preprocessing is fixed.
- **Views.**
  - `full`: the whole frame.
  - `zone`: the estimated operator zone from the existing stage-4 train cache. The zone comes from
    `train_probe.features()` (YOLOX persons → `estimate_zone`, label-free, available at inference), expanded ×1.5
    about its centre and clipped to the frame.
  - If the zone is None, the zone view equals the full view for that clip. Such clips are counted as `zone_fallback`
    and never excluded (§0).
- **Text probes** (`stage5_encoder.PROMPTS`, 8 English prompts, fixed now):
  - The prompts: cashier counting banknotes; hand holding paper money; handing money across a counter; open cash
    drawer; money into a pocket; holding a mobile phone; two people behind a counter; empty cashier workplace.
  - Text embeddings are computed once from the text tower (max_length 64, padding max_length) and L2-normalised.
  - The prompts are a **prior guess about cashier semantics**, not knowledge of the labels. The labelled behaviour
    is unknown to the designers, by the data boundary.

## 2. Per-time-step features (one step per sampled second)
For view v and family F, the base sequence z_t (d = 8) is:
- **F = E (image embedding):** z_t = PCA₈(e_t). The centring and the top-8 components are fitted **on all time steps
  of the training clips of the fold** (unsupervised, but inside the fold).
- **F = T (text probes):** z_t = [cos(e_t, u_p)]_{p=1..8}. Nothing is fitted except the standardisation below.

Window W = 5 s, ending at t. A step is valid if its window holds at least 3 samples. Features, 18 dims:

| Name | Definition | Dims |
|---|---|---|
| level | mean of z over the window | 8 |
| residual | level − median of z over the whole clip (scene normalisation; label-free; whole clip is available at inference) | 8 |
| change | window mean of ‖e_s − e_{s−1}‖₂ on raw normalised embeddings (semantic change rate, not pixel motion) | 1 |
| dist-to-median | window mean of ‖e_s − median_clip(e)‖₂ | 1 |

Standardisation: the mean and std are fitted on the training-fold steps. A std of 0 maps to 1.

## 3. Per-step model, clip score, localisation
- **Model.** L2-penalised logistic regression, fitted by IRLS (numpy, deterministic, intercept unpenalised).
  - Objective: Σ w_i·logloss_i + (λ/2)‖β‖².
  - Class weights make the total positive weight equal the total negative weight, and the weights sum to N_steps.
- **Targets inside a training fold.**
  - Positive steps: steps of positive training clips with t ∈ [t_label − 10 s, t_label + 2 s], for any label time.
  - Negative steps: all valid steps of negative training clips.
  - Other steps of positive clips are **unused** (unknown; no invented negatives or onsets).
- **Clip score.** s(clip) = max over valid steps of the per-step logit. A clip with no valid step is excluded and
  counted.
- **Localisation** (label-free; a hyper-choice, §4):
  - `peak`: the earliest argmax step.
  - `run_end`: the last step of the maximal contiguous run of valid steps that contains the argmax and has
    logit ≥ max − 1.0. This handles long cues that end at the label (lesson S4-2). Window features already end at
    t.

## 4. Fixed hyper-grid (24 configurations; 12 distinct model fits)
| Dimension | Values |
|---|---|
| view | full, zone |
| family | E (PCA₈ of image embedding), T (8 text probes) |
| λ | 1, 10, 100 |
| localisation | peak, run_end |

Nothing else is tuned. W = 5 s, d = 8, the target span [−10, +2] s and the run margin 1.0 are fixed above.

## 5. Nested clip-level leave-one-out (unit = sha16; 17 outer × 16 inner)
For each outer clip k, with Tr = the 16 other clips:
1. **Inner LOO.** For each fit-config g (view, family, λ) and each j ∈ Tr:
   - fit PCA (E only), the standardisation and the logistic model on Tr \ {j};
   - score j, which gives the held-out score s_g(j) and the held-out localisation under both rules.
2. **Inner verdicts.** For each j:
   - θ_{g,−j} = `fit_thr` (maximise clip F1; candidates = scores; ties → higher) over the held-out scores of Tr \ {j};
   - verdict_j = s_g(j) ≥ θ_{g,−j}.
   - Caveat: the models that scored the other inner clips were trained with j's label. This is inner-selection
     optimism only; the outer clip k is never involved.
3. **Selection by classification AND localisation** (lesson S4-1):
   - J(c) = inner clip F1 + inner hit rate. The hit rate is the share of inner positives with |t_pred − t_label| ≤ 5 s,
     regardless of verdict.
   - Ties are broken by inner AUC of the held-out scores, then by the lowest configuration index.
   - The choice is c\* = (g\*, loc\*).
4. **Refit.** PCA, standardisation and the model g\* are refitted on all 16 clips of Tr.
   - θ\* = `fit_thr` over the 16 inner held-out scores of g\*.
   - Outer verdict(k) = s(k) ≥ θ\*; t_pred(k) under loc\*. **One outer prediction per clip.**
5. **Label isolation.** The labels and label times of k are never read for k's prediction. Inputs are the clip's own
   embeddings, its estimated zone (label-free) and the fitted fold objects.
   - Implementation: the models trained on all \ {k, j} are shared by the pair (k, j) and cached by the unordered
     pair. PCA and standardisation are label-free and cached by training set.
   - A perturbation test (as in crit-stage4) must show 0 changes of k's prediction when k's own label or time
     changes.

## 6. Metrics, baselines, controls (outer folds only)
- **Clip level:** TP / FP / FN / TN, precision, recall, F1 over the 17 outer verdicts.
- **Timestamp-endpoint hit rate:** the share of the 5 positives with min_l |t_pred − t_l| ≤ 5 s, regardless of verdict.
- **Event level:** `tools/evaluate.py` `match_events` / `_prf` at ±5 s, through `train_probe.metrics`. Each
  predicted-violation clip emits one event at t_pred.
- **Always-positive:** clip P / R / F1.
- **Random-time:**
  - the analytic expected hit rate, uniform over [0, d] (clipped union of ±5 s);
  - **also** the expected hit rate uniform over the valid scan support (S4-4).
  - The comparison uses the **larger** of the two.
- **Duration-only (nested):** the score is ±duration. The sign and threshold are chosen inside the folds exactly as
  in the stage-4 nested scheme. Clip P / R / F1.
- **Permutation control of the WHOLE selection:**
  - **n = 1000, fixed now from the measured synthetic timing** (§11): a full nested run takes p50 0.36–0.41 s, and the
    first run, which builds the caches, takes 8–10 s. Projected time ≈ 7 min, well under 45 min. Seed 0. Labels are shuffled, and relative label positions are carried over to the receiving clip's
    duration, as in stage 4.
  - Each replicate reruns §5 entirely: PCA and standardisation (label-free, cached), all model fits, inner
    thresholds, the inner selection J, refit, outer thresholds.
  - **Global wall-clock guard.** One overall budget of 90 min (`--budget-total-sec 5400`) covers embedding, the
    nested run, permutations, sensitivity and ablation. When the remainder is insufficient, that phase **stops** and
    is reported `incomplete: <phase>`.
    - n_perm is never adapted to results.
    - If the primary permutations are incomplete, criteria 3 and 5 are "not evaluable", which means **NO-GO by rule**.
  - p50 / p95 and p_perm = (1 + #null ≥ obs) / (1 + n) are reported for outer F1 and for the outer hit rate.
- **Duration confound:**
  - the point-biserial r and AUC of duration against the label;
  - the Spearman |ρ| between the chosen fit-config's 16 inner held-out clip scores and those clips' durations,
    computed per outer fold. Its median and max over the 17 folds are reported, and the median is used by
    criterion 6.
- **Coverage:** frames and steps per clip (min / median / max), zone-None count, failures, label times ≤ duration,
  and the chosen-configuration counts per dimension.

## 7. Decision rule (primary analysis, all 17 units)
**GO** only if all of the following hold:
1. outer clip F1 > always-positive F1;
2. outer clip F1 > **duration-only nested F1** (S4-4);
3. outer clip F1 > permutation p95 of F1;
4. outer hit rate > max(random-time uniform, random-time over valid support);
5. outer hit rate > permutation p95 of the hit rate;
6. median |ρ|(chosen score, duration) < 0.5. Otherwise the result is flagged duration-dependent and is not a GO;
7. the primary set is complete: none of the 17 units failed decode or embedding (§0).

Anything else is a **measured NO-GO** for this encoder, feature family and procedure on this sample. A GO only
licenses proposing **one** locked holdout evaluation of the frozen procedure. Independent critical review must
approve it first.

Even a GO keeps the holdout closed until a separate agreement (§0).

**Prior expectation.** NO-GO is likely.
- The audit answer V1 = 4/5 "unseen" **constrains interpretation only**. It does not prove that the labelled
  behaviour is invisible, and it does not explain any outcome.
- n = 5 positives gives low power (S4-3: GO ≈ 93 % only for a near-perfect cue in 5/5).
- One hit is worth 0.2.

## 8. Sensitivity (not deployable, not used for the decision)
- The view is fixed to `full`: the grid is reduced to family × λ × loc, with the same nested scheme and n_perm 200.
  This shows whether the zone crop drove the choice.
- Zone-agree ablation (R1 half vs zone centre, as in stage 4 §6; a zone-fallback clip counts as a mismatch), with
  n_perm 200. It uses human R1, so it is labelled ABLATION.

## 9. Synthetic self-test (before the real run; LOCAL_PRIVATE_DIR in scratch)
- Fake C02 split, labels and inventory map in scratch. The holdout files are created with mode 000, and an `open()`
  spy asserts that no holdout path is opened. Outputs are grepped: no sha and no "/" may appear.
- **Scenarios** (make_synthetic clips, 17 units, 5 positives, durations 15–60 s):
- **Scenarios.** Rendered clips: per-clip background and counter colours, and a distractor client figure for 6 s in
  every clip. One clip has no zone cache (expect `zone_fallback` = 1).
  - **planted:** a large green box with white "banknote" stacks in the zone during the 8 s before the label, in
    positives only. Must give GO.
  - **null:** no cue. Must give NO-GO.
  - **long:** a 30 s cue ending at the label. Must give hit ≥ 0.8, with `run_end` chosen in ≥ 9/17 folds.
  - **corrupt:** one train negative truncated to 40 %. Must give `decode_incomplete` = 1, no cache, the unit counted
    as failed, and no GO ("INCOMPLETE PRIMARY").
  - n_perm 200 (20 for corrupt).
- **Power (bounded; run after the real run; does not affect the decision).**
  - The measured cue direction (mean cue embedding − mean negative embedding, planted fixture) is added to the null
    fixture's embeddings during the 8 s before re-drawn label times, for 5/5 and for 4/5 positives.
  - 10 replicates per level with n_perm 100, budget 20 min. The GO rate is reported as rough power. It is not
    evidence about real data.
- **Perturbation test** of §5.5. Each outer clip's label is flipped and its time shifted, which must give 0
  prediction changes. An inner-label flip must change other clips.

## 10. Licences (primary sources, retrieved 2026-10-17/18; not a legal opinion)
| Component | Pin | Licence (quoted) | Primary source | Notes |
|---|---|---|---|---|
| SigLIP2 B/16-224 weights | HF rev `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | card YAML `license: apache-2.0`; the repo has **no LICENSE file** | https://huggingface.co/google/siglip2-base-patch16-224/resolve/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2/README.md | not gated, no remote code. **Training data: WebLI (unreleased, Google-internal); data terms UNVERIFIED.** Apache-2.0 on the weights does not certify the data |
| big_vision (SigLIP2 reference code) | commit `0127fb6b337ee2a27bf4e54dea79cff176527356` | "Apache License Version 2.0, January 2004"; README: "released under the Apache2 license" | https://raw.githubusercontent.com/google-research/big_vision/0127fb6b337ee2a27bf4e54dea79cff176527356/LICENSE | not executed; reference only |
| transformers (SigLIP implementation) | 5.20.1 | "Apache License Version 2.0" | https://raw.githubusercontent.com/huggingface/transformers/v5.20.1/LICENSE | SigLIP is native (`models/siglip`) |
| torch (export and smoke only) | 2.14.1 | BSD-style ("From PyTorch: Copyright (c) 2016- Facebook, Inc …") | https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/LICENSE | research venv only |
| onnx / onnxscript (export) | 1.23.1 / 0.7.2 | "Apache License Version 2.0" / "MIT License" | raw LICENSE at tags v1.23.1 / v0.7.2 | research venv only |
| onnxruntime (inference) | 1.31.0 | MIT | already in LICENSES.md | main venv, already pinned |

Weight files:
- `tools/fetch_weights.py siglip2_base_patch16_224` pins 8 files with size and SHA256.
- `model.safetensors`: 1 500 800 904 B, sha256 `612923381c76ec5a…`. `tokenizer.json`: sha256 `cb9140fa…`.
- Verified: "8 files sha256 ok".

Rejected alternatives:
- **openai/clip-vit-base-patch32.** No HF licence tag. The card says "**Any** deployed use case of the model —
  whether commercial or not — is currently out of scope."
- **laion/CLIP-ViT-B-32-laion2B.** The card is MIT, but repeats the "out of scope" deployed-use clause and
  recommends against "ready-to-go industrial products". LAION-2B also carries data concerns.
- **Florence-2.** It requires trust_remote_code.
- **DINOv2-S/B** (Apache-2.0 at HF rev `ed25f3a3` / `f9e44c81`; GitHub LICENSE Apache-2.0). It is acceptable, but
  not chosen (§12).

Install pins, research venv only (coordinator adds them to requirements-research.txt): torch==2.14.1, onnx==1.23.1,
onnxscript==0.7.2. The transitive additions, all at least 2 weeks old, are ml-dtypes 0.6.0, onnx-ir 1.0.0,
protobuf 7.36.2, setuptools 84.0.0, networkx 3.7, sympy 1.14.0 and mpmath 1.3.0. No existing package changed.

## 11. Budget smoke (SYNTHETIC frames only; Apple M5, 10 cores, 16 GB)
| Item | Measured |
|---|---|
| install (uv, research venv) | 94 s |
| weights (1.4 GB; resumed and SHA256-verified) | about 3.5 min in total |
| torch load / text tower (8 prompts) | 0.59 s / 1.05 s |
| torch CPU ms/frame | bs1 51.3, bs8 29.6 |
| torch MPS ms/frame | bs1 21.5, bs8 15.0 |
| ONNX export | 2.5 s |
| onnxruntime CPU ms/frame | bs1 35.6, bs8 32.1; load 0.18 s; cos vs torch 1.000 |
| onnxruntime CoreML | bs1 70.7 ms; bs8 **failed** (dynamic batch). Not used |
| peak RSS | torch child 2.0 GB; ORT child 2.1 GB |
| model math | IRLS logistic (400×18) 0.11 ms/fit; PCA SVD (400×768) 17 ms |

Projection for the real train pass:
- 657 s at 1 fps is about 660 frames, ×2 views = 1 320 embeddings × 33 ms = **about 45 s** of inference.
- Sequential decode of about 20 k source frames: **about 1–3 min**, not measured on real data.
- Total **about 5 min**, peak **about 2.5 GB**, cache **about 2 MB**.
- Measured on the synthetic 17-unit fixture (~630 frames): embedding wall 65–72 s for both views. Full nested run:
  p50 0.36–0.41 s, p95 0.40 s, first run 8–10 s. Real run projection: embedding ~5 min, then 1000 + 2 × 200
  permutations at ~0.4 s ≈ 10 min, so **~15–20 min**. Self-test ~6 min. Power ≤ 20 min. Everything fits the 90-min
  budget.
- Memory watchdog proof on the REAL child entry point (`--child embed`, mode memtest):
  - the cap was lowered to 256 MB with a bounded allocation of 300 MB;
  - result: `status=memory_cap`, peak seen 280 MB, 0 leftover children;
  - the production cap is PILOT_MEM_MB = 8192.

Projection for the nested run:
- About 1 850 fits plus scoring per full nested run, **about 1–3 s**.
- 1000 permutations: **about 20–50 min**. The fallback of §6 applies.
- The self-test adds about 3 × 17 synthetic clips: **about 3 min** of encoding plus the power replicates, **about
  10 min**.
- **Total stage-5 compute ≲ 75 min, ≤ 3 GB RAM, < 10 MB of new cache.**

## 12. Why this approach (and what it cannot do)
**Why SigLIP2.**
- It is a contrastive image–text encoder trained on WebLI, a web-scale multilingual image–alt-text corpus. Its
  objective adds captioning-decoder, self-distillation and masked-prediction losses, which aim at "improved
  semantic understanding, localization, and dense features".
- Its embedding space is aligned with language. That allows a **pre-registered, low-dimensional, label-free** probe
  family (T, 8 prompts), which suits n = 17 better than learning a direction in 768-d.
- The image family (E) covers semantics the prompts miss.
- Shop, cash and counter scenes are common web imagery, so generic concepts (money in hand, a person at a desk,
  two people) are plausibly encoded.
- Unlike the stage-4 features (YOLOX COCO counts and motion), it represents *what* is in the frame rather than how
  many boxes there are or how much moves.

**Why not DINOv2.**
- DINOv2 is self-supervised on LVD-142M and has strong dense and instance features, but no language alignment.
  Every semantic direction would have to be learned from 5 positives.
- It remains the natural follow-up if family E wins over T.

**Limits.**
- Single frames at 224 px (the zone crop helps): banknotes and hands from a ceiling CCTV may cover only a few
  patches.
- The CCTV top view is out of the web-photo distribution.
- There is no temporal model inside the encoder; temporal structure comes only from §2's window statistics.
- The prompts are English guesses.
- The audit's V1 4/5 "unseen" constrains interpretation only. It does not prove invisibility, and it is not a cause
  of any outcome. If the behaviour is not visible in frames, no frame encoder can recover it.
- The scene-median residual can cancel a cue that lasts most of the clip; family `level` features keep it.
- With n = 17, any result is fragile. Per §0, no claim beyond this sample is made.
