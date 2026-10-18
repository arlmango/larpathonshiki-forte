# Critical review (wave 4): independent, read-only

Reviewer: critical-reviewer. Scope: app/jobs.py, worker.py, config.py, main.py; tools/annotate.py and inventory.py; tests/;
repo-wide boundary sweep. Second pass: pipeline.py, detector.py, rules_default.json, norms.py, fetch_weights.py (after the
coordinator marked them FINAL). Every experiment used synthetic clips (tools/make_synthetic.py) or synthetic JPEGs in the
scratchpad. No real video, frame, label, file name or scene text was opened or listed. Owners' reports were treated as claims.

## Run numbers
- `pytest -q tests -p no:cacheprovider`: **87 passed, 1 skipped, 0 failed, 24.45 s.**
- `SYNTHETIC_DETECTOR=1 tools/stress_queue.py 24 8`: **RESULT PASS.** A (subprocess): offered 24, admitted 11, 429 13; latency
  0.52–8.06 s; 0 left processing, 0 over the limit, 0 stray files, durable 11/11. B (in-process): admitted 12, 429 12, max
  9.46 s, durable 12/12. In A, 6 of the 11 admitted jobs ended with `time_budget_exceeded`: 2 corrupt, 1 synth and 3 that
  finished normally in B (see M3).
- Live burst (uvicorn, default settings, 60 concurrent uploads of a 20 s synthetic clip): **200 ×6, 429 ×54.** All 6
  admitted jobs ended `failed/no_validated_rule` within 0.7 s. data_dir was clean afterwards and no worker children were left.
- Peak RSS of the worker child on a synthetic 1080p clip of 145 s: **397 MB** with yolox_s (stage 1 only, the figure is
  not a person, so it ends `too_few_detections`) and **235 MB** with the synthetic detector (stage 2 and evidence included).
  Wall/CPU time for yolox_s: 65.9 s wall and 243.6 s CPU (4 ORT threads).

## Findings, by severity

| ID | Sev | Where | Owner | Blocks demo? |
|---|---|---|---|---|
| H1 | **HIGH** | app/worker.py:257-266 (`_limit_cpu`), called at :295 | queue-architect | **YES** for any clip that needs more than about 60 s of wall time |
| H2 | **HIGH** | app/pipeline.py:304-305 (`rule_states`) | detection-architect | no (default ruleset); breaks the F7 integrity claim |
| H3 | **HIGH** | app/pipeline.py:366, :379, :437-438, rules_default.json | detection-architect | no (default ruleset); the coordinator's claim (3) is false |
| M1 | MED | app/pipeline.py:302-310, :418 | detection-architect | no |
| M2 | MED | app/pipeline.py:457 + public schema | detection-architect / coordinator | demo wording only |
| M3 | MED | app/jobs.py:174-176, :131 | queue-architect | **possibly**: a judge's parallel test gets 429 at 7 concurrent jobs |
| M4 | MED | tests/test_pipeline_rules.py (second-person and default-rule negatives) | test-dev | no |
| M5 | MED | in-process cv2 decode: pipeline.export_observations / analyze(observations_dir=…) | detection-architect | no (boundary) |
| M6 | MED-LOW | app/pipeline.py:81, app/detector.py:274 → GET /api/v1/model | detection-architect | no |
| M7 | MED-LOW | app/worker.py:144-147, :189-203 | queue-architect | no |
| L1–L14 | LOW | see below | various | no |

### H1: RLIMIT_CPU counts CPU-seconds across all threads, so a 4-thread child is killed at about 1/3.7 of its budget
`_limit_cpu` sets `RLIMIT_CPU = (deadline - now) + 10` **CPU** seconds. onnxruntime runs `DETECTOR_THREADS=4` and uses about
3.7 cores, so SIGXCPU kills the child after roughly 65 s of wall time with the production budget. The parent sees
`rc = -24`, which is not SIGKILL, and reports `worker_crashed`.
Repro (synthetic): `make(hd145.mp4, 145, fps=25, size=(1920,1080), absent=(40,70))`, an opt-in ruleset, then
`DETECTOR_MODEL=yolox_s ALLOW_UNVALIDATED_RULES=1 JOB_SETTINGS=… JOB_DEADLINE=$((now+40)) python -m app.worker in out`
→ **rc=152 (128+SIGXCPU), no out.json**. With `JOB_DEADLINE=now+220` the child also ended abnormally ("command terminated
abnormally") after 61.9 s wall / 227 s CPU. Without a deadline it completes: 65.9 s wall / 243.6 s CPU, and that is stage 1
alone. Real clips also run stage 2 (up to 60 s of video at 8 fps, about 480 more inferences), so they will exceed about 239
CPU-s every time. Expected effect: long real clips with yolox_s → `worker_crashed`.
Fix: remove RLIMIT_CPU. The parent's SIGKILL, the watchdog and `_watch_parent`'s deadline+30 exit already cover this. If
the limit stays, scale it: `remaining * (threads + 1) + margin`. Re-run the 145 s / 1080p measurement after the fix.

### H2: the "validated" gate only checks syntax, so 0, -1, NaN, inf and 1e9 all count as a measured holdout value
`validated = status=="validated" and validated is True and isinstance(holdout.value, int|float)`. Results with no opt-in, on
the synthetic absence clip: `holdout.value` = 0, -1, NaN, inf, 1e9 or 0.95 → **done/violation**, `unvalidated_marker=False`,
`verdict_basis='validated rules'`. A "fake metrics" dict `{"value":0.99,"n_clips":0,"metric":"made_up"}` is accepted.
These are correctly refused: the string "1", True, None, `validated:"true"`, `status:"Validated"`, `enabled:1`.
Consequence: one JSON edit removes the "НЕ валидировано" marker from public descriptions. Nothing ties the number to
tools/evaluate.py output.
Fix: require a finite value in (0,1], `n_holdout_clips >= min_holdout`, a metric name, and ideally a hash of the
evaluate.py report. Otherwise raise `invalid_config`.

### H3: the coverage gates can be switched off from the rules JSON, which gives a blind `no_violation`
`min_person_sample_ratio`, `zone.min_occupancy_to_analyse`, `zone.override` (or the `OPERATOR_ZONE` env var),
`min_confidence_to_emit` and `max_events` all come straight from the JSON with no floor. Repro uses EmptyDetector (detects
nothing) and a rule "validated" with holdout 0.9:
gates intact → `too_few_detections` (OK); ratio=0 → `zone_unreliable`; ratio=0 plus override → `zone_unreliable`;
**ratio=0, override and min_occ=0 → done/no_violation, basis "validated rules"**; ratio=-1 and min_occ=-1 on the absence
clip → no_violation; `OPERATOR_ZONE=0,0,1,1` with the gates at 0 under opt-in → no_violation; `min_confidence_to_emit=2`
or `max_events=0` on the absence clip → no_violation.
Fix: hard-coded floors (`max(json, 0.10)` / `max(json, 0.15)`), schema bounds checked on load (`min_confidence_to_emit` in
[0,1], `max_events >= 1`) → `invalid_config`.

### M1: an unknown rule key counts as an "active" rule
`rules["rules"]["zz"]={"enabled":true}` under opt-in, or the same rule with a fake validated holdout (value 0) and no
opt-in, passes the `no_validated_rule` gate. Nothing detects "zz", so the result is **done/no_violation**.
Fix: `active` must be a subset of the detectable, verdict-eligible types (`{UNATTENDED}`); unknown keys → `invalid_config`.

### M2: under the opt-in, a `no_violation` result says nothing in public about being unvalidated
`ALLOW_UNVALIDATED_RULES=1` with an operator present the whole time → `{"verdict":"no_violation","violations":[]}`. The
"UNVALIDATED" basis exists only in the stripped `meta`. The only signal is the UI banner (static/index.html:46 via
/api/v1/model). API clients get a no_violation from unvalidated rules with no marker, which is what F7 forbids. This needs a
coordinator decision: fail with a code under the opt-in, or document it as demo-only and keep the banner.

### M3: admission control only learns from `done` jobs, and a busy slot is assumed to free up within 1 s
The EWMA is updated only when `status=="done"` (jobs.py:174). With the default ruleset every job is `failed`, so the
estimate stays at the 60 s floor forever. Result: 60 concurrent uploads → only **6 admitted (capacity 22)**, even though
each job took 0.3 s. `QUEUE_LIMIT=20` has no effect.
The other way round, `remaining = max(est - elapsed, 1.0)` (jobs.py:131) assumes that a job running past `est` frees its
slot in 1 s. A hung or slow job blocks its slot until the deadline, so jobs behind it are admitted and then time out (the
stress run above: corrupt/synth/ok jobs ended `time_budget_exceeded`).
Fix: update the EWMA from every completion that was not killed; count killed jobs as the full budget. Estimate a busy slot
as `max(est - elapsed, small)` but let it grow toward the deadline, i.e. `deadline - now` once elapsed exceeds est.

### M4: the strict negatives partly pass vacuously
- `test_optin_second_person_rule_cannot_produce_violation`: the synthetic `two=(5,35)` clip produces **0 SECOND
  observations**. With `pipeline.NEVER_VERDICT = set()` and `observation_only=false` (simulating a regression that deletes
  the guard) the result is still `no_violation`, so the test cannot detect that regression. It needs a clip or detector
  stub that actually produces the SECOND observation; assert it appears in `observations`, then assert it is not a
  violation.
- `test_default_rules_zero_persons_with_counter_motion_is_failed` and `test_default_rules_two_persons_is_failed_not_violation`
  fail at the rule gate (pipeline.py:420) before `_observe`, so they do not exercise F1 detection logic. F1 is really
  covered only by `test_empty_detector_with_optin_is_failed_not_no_violation`.
- No tests for H2, H3 or M1 (NaN/0 holdout, dummy rule key, gate floors in the JSON).

### M5: native OpenCV stderr carries the full file path (boundary issue for in-process decoding)
Repro: a garbage file `…/canarydir/CANARY_SECRET_NAME.mov` passed to `cv2.VideoCapture` prints
`OpenCV: Couldn't read video stream from file "<full path>"` and `[mov,…] moov atom not found` on stderr.
`OPENCV_LOG_LEVEL=OFF` and `OPENCV_FFMPEG_LOGLEVEL=-8` do **not** suppress it. What is safe and what is not:
- The API path is safe: the child has stdio=DEVNULL.
- `tools/inventory.py --probe` is fixed: an isolated child with stderr=DEVNULL; 0 canary hits on stdout or stderr (verified).
- **`pipeline.export_observations()` and `analyze(observations_dir=…)` decode in-process.** "Prints nothing" is true for
  Python, but not for native stderr.

Fix: run these in a child with stderr=DEVNULL, or `os.dup2` fd 2 to /dev/null around the decode.

### M6: GET /api/v1/model returns raw exception text
`rules_error = str(e)` (pipeline.py:81). Repro: RULESET pointing at a missing file → `"[Errno 2] No such file or
directory: '<full path>/SECRET_rules_dir/r.json'"`. `detector.describe` puts `"error": str(e)` in the response
(detector.py:274), which includes the weights basename and a hint. The endpoint requires the key, but this still breaks
"static codes, never paths". Fix: put the `DetectorError.code` and a static `rules_error: "invalid_config"` in the
response; keep the message out of it.

### M7: the F10 validator accepts a truncated JPEG if EOI is re-appended
Synthetic 160×120 noise JPEGs: truncated to 60 % plus `FFD9` → **accepted**; truncated to 10 % plus `FFD9` → **accepted**;
JPEG, junk bytes, then `FFD9` → accepted. libjpeg prints "Corrupt JPEG data: premature end of data segment" to the
parent's stderr; it contains no content. The F10 addendum says "truncated … ⇒ invalid_result". The practical risk is low
because the producer is `cv2.imencode`. Fix: a strict decode that rejects libjpeg warnings (for example Pillow `load()`
with `LOAD_TRUNCATED_IMAGES=False`), or document this as a known limit. Everything else in the F10 suite works; see the
verified list below.

### Low
- **L1** main.py:75: `_Guard` applies only to the exact `POST /api/v1/analyze`. Any other path or method (trailing slash →
  307, 404, 405, GET with a body) has its body drained by uvicorn without auth or timeout: 5 MB accepted, socket held
  until the client closed it. No disk or memory growth observed. The 307 `Location` reflects the Host header
  (`http://evil.example/api/v1/analyze`). Fix: in the guard, answer any request with a body on an unguarded route with a
  static 413/404 and close; set `redirect_slashes=False`. Owner: api-dev.
- **L2** main.py:79-80 vs :147: with a duplicate `X-API-Key`, the guard uses the last value and FastAPI the first.
  `[bad,good]` passes the guard and the body is read up to the cap before 401. This is not a bypass, because it needs the
  key. Owner: api-dev.
- **L3** main.py:136-137: the timeouts are read from env with `_env_f` and not from `Settings.receive_timeout_sec` /
  `upload_timeout_sec`, which now exist. Values set in Settings in code are ignored, and the config.py check `<300 s` is
  not what main.py actually uses. Owner: api-dev.
- **L4** Starlette 400 texts are not uniform ("Too many fields. Maximum number of fields is 1000."). They are static and
  echo no input. Informational.
- **L5** data_dir not writable → `500 {"detail":"internal error"}` (static, OK). uvicorn prints a full traceback to stderr;
  it contains the data_dir path and a uuid name, no real-data content. Acceptable.
- **L6** tools/annotate.py: (a) CSRF: `POST /api/c/C01/split` with `Content-Type: text/plain`, which a browser sends
  without preflight → **200**, and the attacker picks ratio and seed. `request.json()` ignores the content type. Fix:
  require `application/json` and check `Origin`. (b) `list_clips` (:43-44) uses `isfile`, which follows symlinks: a
  symlink to a file outside the candidate dir is listed and served (206). inventory skips symlinks, so the counts and
  indices disagree. (c) An I/O error on a real file goes to uvicorn's error log as a traceback with the path. Owner: ui-dev.
- **L7** `_atomic_write` leaves the temp file behind on an exception; inventory `write_map` is not atomic and does chmod
  after the write (a 0644 window). Owner: ui-dev.
- **L8** detector.py:241-242: an exported `SYNTHETIC_DETECTOR=1` silently switches production to the colour detector.
  Only /api/v1/model shows it. Fix: `create_app` refuses to start without an explicit test flag. Owner: detection-architect
  / api-dev. The conftest autouse setting is fine for tests.
- **L9** tests/test_contract.py: `test_dimension_bounds` is still skipped, but bounds now exist (worker.py:112, :202).
  My check: 1×1 and 15×15 → invalid_result, 16×16 accepted, 4097×16 → invalid_result. Add these fixtures. Owner: test-dev.
- **L10** .gitignore: `*.mov`/`*.mp4` match `A.MOV` only because macOS sets `core.ignorecase`. Add `*.MOV`, `*.MP4`,
  `*.avi`, `*.mkv`. Owner: coordinator.
- **L11** Uploads are spooled in `$TMPDIR` (Starlette SpooledTemporaryFile, anonymous) before the copy into data_dir.
  Concurrent uploads are unbounded (no `limit_concurrency`), and admission runs only after the full upload. Owner: api-dev.
- **L12** By default 2 workers × 4 ORT threads = 8 threads, on a machine with 10 cores of which 4 are P-cores. A single
  yolox_s job takes about 66 s for stage 1 on 145 s at 1080p; two at once will be slower. This adds to M3.
- **L13** When the deadline is close, stage 2 is skipped completely (`truncated=True`) and a violation from stage 1 alone
  is still emitted (`refined=False`, ×0.8). The flag exists only in meta. This is a design note.
- **L14** FastAPI 0.143 telemetry: only opentelemetry-api is installed (no SDK or exporter), so it does nothing. It would
  export only with an SDK installed and `FASTAPI_OTEL_AUTO_CONFIGURE=true`. Informational.

## Verified OK (by experiment or line-by-line reading)
- **Deadline**: the budget starts at the request (guard `scope.state.t0` → `submitted_at`), including queue wait. Hung FIFO
  children are SIGKILLed as a process group at the deadline (test: 3 s; stress: 8.0 s); the watchdog is the backup. The
  final status is set once (`_finish`), and the processing marker cannot overwrite a final record.
- **Admission**: 429 removes the upload (tested). Cleanup touches only files matching `_TEMP_RE`/`_RESULT_RE`.
- **Two stores on one data_dir** (experiment): B sees A's processing and done records, and a new store's sweep does not
  delete A's in-flight upload.
- **Parent SIGTERM** (uvicorn): lifespan `close()` runs, the record becomes `failed/service_shutting_down`, the child is
  gone and the FIFO upload is removed. **Parent SIGKILL**: the child exits within 2.5 s through the ppid watch, even while
  blocked in a FIFO `open`. The marker stays `processing`, then becomes `job_lost` after budget+grace+5. The orphan upload
  is swept after 10 min.
- **F10**: `cv2.imdecode` actually decodes; decoded size must equal the SOF header; 16–4096 px; 101 B–2 MB; strict base64;
  1–3 frames; finite ≥0 numbers; no_violation⇔empty. Non-public keys are stripped by re-projection in the in-process path
  (`_done`), the child, the parent re-validation, and when reading from disk. A worker.json with a path in `error` →
  `analysis_failed`; status "Done" → `invalid_result`; extra keys never reach the output.
- **Errors**: closed allow-list (`safe_code`); child stdio is DEVNULL; logs carry only job_id, rc, elapsed, errno or class
  name. No `_tail` and no stderr anywhere in app/.
- **F8 (live uvicorn)**: bad key → 401 after ≤1 chunk; Content-Length over the cap → 413 before the body; chunked over
  the cap → 413 early; CL mismatch, CL+TE, negative or non-numeric CL → uvicorn 400; `%61nalyze`, `?x=1` and `#f` are still
  guarded; `/API/…`, `//api…` and `/./` → 404; HEAD/OPTIONS → 405; `Expect: 100-continue` with a bad key → 401 with 0 body
  bytes; 408 on stall and on total time (tests); data_dir empty after all cases; 500 body static.
- **Boundary sweep** (`print(`, logging, `str(e)`, traceback, listdir, glob): the only problems are M5, M6 and L6(c).
  tools/inventory now uses an extension allow-list and an isolated probe (verified with the canary). annotate prints only
  counts; Host/client check → 403 on a rebinding Host; Range `-0` → 416; the cid regex plus the map lookup prevents path
  traversal.
- **Repo**: no file has more than 500 lines (largest: pipeline.py 486). .gitignore covers data/, local_private/,
  weights/, *.mov, *.mp4. The only network code is tools/fetch_weights.py (urllib). static/ has no external URLs.
- **Second pass**:
  - (2) `second_person_in_operator_zone` never produced a violation in 3 attempts: enabled+validated+observation_only=false
    without opt-in; opt-in with `observation_only` removed; both rules enabled. Caveat: the synthetic clip never produces
    the observation (M4).
  - (4) Deadline: `_check` runs on every sampled stage-1 frame, every stage-2 frame and every evidence frame. A 2 s budget
    → `time_budget_exceeded` at 2.01 s. Stage 2 shrinks to the remaining time (6–9 s budget → windows skipped; 30 s → both
    windows run).
  - (5) `export_observations` and `observations_dir` are not referenced by main.py or worker.py.
  - (6) SHA256 is checked before `InferenceSession`: a tampered `yolox_nano.onnx` → `model_invalid` with 0 sessions
    created. A file under a custom basename skips the hash check by design (license "unknown"). A junk custom file →
    `model_load_failed`.
  - (7) No network calls in pipeline or detector.
  - (8) Evidence frames carry only the drawn zone and boxes (no face processing). Descriptions say "Наблюдаемо / Не
    проверено", "Кандидатные нормы … (не обоснование вывода)" and "вывод не может считаться установленным нарушением", and
    the "НЕ валидировано" marker when the rule is unvalidated (it disappears under H2).
  - (9) YOLOX decoding matches the official `demo_postprocess` and `multiclass_nms` (class-agnostic): strides 8/16/32,
    `meshgrid(x,y)`, `(xy+grid)*s`, `exp(wh)*s`, obj×cls score, argmax==person(0), NMS 0.45, letterbox top-left with pad
    114 in BGR 0–255 without normalisation, `/ratio`. The output shape check is 85 channels.
  - (10) Memory: samples keep only boxes plus a float16 48×27 grid; no frames are retained; evidence is decoded again.
    Peak RSS 397 MB / 235 MB, as above.

## Not verified
- Real person-detection quality of YOLOX. The synthetic figure is not detected (`too_few_detections`), and real data is
  off-limits.
- Linux-only behaviour: RLIMIT_AS, the OOM path, case-sensitive .gitignore.
- Slow-loris on the header phase (uvicorn level, outside the in-app F8 scope).
- uvicorn `--workers N`: by reasoning, admission is per process and status works across processes through the shared disk
  (the two-store experiment supports this).
- Whether unblurred customer faces in public evidence frames are acceptable is a policy question; the code does no
  identification.

Model I believe I run as: **claude-opus-5-5 (Opus 5.5)**. This is a self-report from my system prompt, not verified.

## Pass 3: re-check of H1, M3 and #8 (synthetic only)
I did not re-check H2, H3, M1, M5 or M6; the cross-reviewer has them. M7 the user accepted as a known limitation.
Load during these runs was high (load average 11.9, other agents were running stress), so wall times are upper bounds.

| Item | Verdict | Evidence |
|---|---|---|
| **H1** RLIMIT_CPU | **CONFIRMED fixed** | app/worker.py:283 (comment, no setrlimit CPU). The old repro (`JOB_DEADLINE=now+40`, yolox_s, 1080p 145 s): **rc=0**, `failed/time_budget_exceeded` at 39.3 s through the soft deadline, where it used to exit 152. `JOB_DEADLINE=now+220`: **rc=0**, 130.7 s wall / **270 CPU-s**, more than the old ~239 CPU-s that killed it before; RSS 396 MB. |
| Hung FIFO killed at budget | **CONFIRMED** | `test_hung_child_fifo_killed_within_budget` passed; stress phase A: hang → `time_budget_exceeded` ×4, max latency 8.07 s for an 8 s budget, 0 over the limit. |
| **M3** admission | **CONFIRMED fixed, one residual** | jobs.py:117-120 (max of the last 20), :135 (an overrunning job holds its slot until its deadline), :177 (recorded unless the job hit the budget). Live burst of 60 from a foreign CWD: **cold 8 accepted / warm 22 accepted** (= capacity); all final (`no_validated_rule`, which is expected with default rules); 0 leftovers; 0 children after shutdown. |
| **#8** CWD-independent paths | **CONFIRMED** | config.py:55-58. From `scratch/review/foreign` with RULESET and DATA_DIR unset: data_dir → `<ROOT>/data/jobs`, ruleset → `<ROOT>/app/rules_default.json` (exists); relative `RULESET=app/rules_default.json` → `<ROOT>/…`; `dataclasses.replace` keeps absolute paths. uvicorn `--app-dir` from the foreign CWD: the child worker runs, the rules load (2 rules, jobs reach `no_validated_rule`). |
| pytest | **131 passed, 0 failed, 0 skipped** (53.9 s) | as expected |
| stress 24/8 | **PASS** | A: admitted 14, 429 ×10, 0 stuck, 0 over the limit, durable 14/14. B: admitted 24/24, durable 24/24, max 9.55 s. |

**Residual R1 (MED-LOW, queue-architect, config.py:41 / jobs.py:143): the 20 s margin is fixed and applies to the whole
budget.** With the stress budget of 8 s, a 20 s margin turns admission off: phase B admitted 24/24 and 20 of them ended
`time_budget_exceeded`; phase A admitted 14 and 10 timed out. The hard deadline still holds (0 over the limit). Make the
margin proportional (for example `min(20, 0.1*budget)`), and set `admission_margin_sec=0` in tools/stress_queue.py so the
stress run actually exercises admission.

**Residual R2 (LOW, design): the estimate ignores clip length.** After fast failures the warm estimate is 0.40 s, so a
following burst of real 145 s clips would be admitted up to capacity (22) and most of them would time out. Use a
duration-aware estimate: seconds per second of video × the probed duration.

**Demo risk, not a bug:** under load, stage 1 alone for a 145 s 1080p clip with yolox_s took 130.7 s wall (65.9 s on the
quieter machine earlier). Stage 2 and evidence come on top of that, and with 2 concurrent jobs (2 × 4 ORT threads) a long
1080p clip can hit `time_budget_exceeded` (230 s). For the demo: WORKERS=1 or DETECTOR_THREADS=2 per worker, or
yolox_tiny / SAMPLE_FPS=3, and time it once on a quiet machine. **Nothing found in H1/M3/#8 still blocks a demo.**

## Stage 2: methodology and leakage review of tools/train_probe.py and detector.detect_classes()
Read-only. The probe was run only on synthetic data in the scratchpad: 17 train clips (5 positive, 12 negative) plus 4
holdout clips set to mode 000, in two variants, one with a planted operator-absence signal and one without. The real
`local_private/train_probe/summary.json` was checked for structure only; no values were printed or quoted.
pytest: **131 passed**.

### Verified OK
- **Holdout isolation (main path).** `train_units` iterates only shas with `side=="train"`
  (train_probe.py:132) and re-hashes every opened clip (:138). With all holdout clips at mode 000 the full run
  finished with rc=0, so no holdout file was opened. Cache keys are `<train sha16>_<fps>.npz` (:399). The cache holds
  only numeric arrays: `t`, a float16 48×27 motion grid, `dets` and `dur` (:75).
- **Unit of observation = the clip (one distinct sha16).** Duplicates collapse through `idx_of.setdefault` and the
  sha-keyed split. A1 is paired per positive clip (:282-299), and its null uses random times inside the same clips.
  A2 and B are clip-level, the label permutation is over clips (:333-336), and the reported n is the number of clips.
  Temporal samples are never pooled.
- **Nested selection.** In `loo(None)` (:344-358), feature, transform, direction, window and threshold are all chosen
  by `fit_threshold` and AUC on `tr_` only, i.e. without the held-out clip. `clip_score` uses only the clip's own
  samples and its own median (:245-251), so nothing crosses clips. There are 22 features × raw/dev × ±1 × {4, 10} s =
  **176 configurations** (:323). The permutation floor (:333-336) applies the same best-of-176 max-AUC selection as
  the observed best AUC, so the comparison is like with like. The full-train variants are labelled
  `OPTIMISTIC_EXPLORATORY` in both the summary and stdout (:367-372, :449, :462).
- **Event metrics.** They reuse `tools/evaluate.py` `match_events` and `_prf` (:269), with ±5 s (`TOL`), and every
  label time counts toward FN (:276). Extraction failures and missing zones are excluded and counted in
  `coverage.extraction_failed` (:404-411), never as no_violation.
- **detect_classes() (app/detector.py:243-266).**
  - It is referenced only by tools/train_probe.py; nothing in main.py, worker.py or pipeline.py calls it.
  - Production `detect()` is unchanged.
  - Person rows come out identical by construction, but I checked this by reading the code, not by running both on
    real persons (no person imagery is allowed). Both use the same argmax==person mask with the person score ≥
    `self.conf`. Persons get class offset 0 in the per-class NMS, so their NMS is the same as in `detect()`.
- **Privacy of outputs.**
  - Synthetic runs: 0 sha, path or clip-name hits in stdout or summary.json, and 0 bytes on stderr.
  - Real summary.json: same key structure as the synthetic one; every string value is a feature or config token; no
    `/`, no media extension, no Cyrillic. The 123 hex-like tokens are all digit-only float mantissas, none contains
    a–f.
  - Real summary.json lists only feature-level aggregates: 22 A1 rows, 15 A2 rows, 3 B1 rows.
- **Printout.** The probe's stdout has table headers only. It makes no semantic claims about the labelled events and
  compares no metrics across types: A2 sits next to the permutation max-AUC (like with like), and B1 is not compared
  to that floor.
- **Sanity (power) check, synthetic.**
  - Planted signal: A1 `zone_empty` pAUC 1.0 (null95 0.905); A2 best AUC 1.000 vs permutation max-AUC p50 0.700 /
    p95 0.833; nested clip F1 **1.000** (always-positive 0.455). The procedure can find a strong signal.
  - No signal: best AUC 0.600 vs p50 0.725. Nested F1 came out **0.500**, which is *above* always-positive 0.455 by
    chance alone.

### Findings
| ID | Sev | Where | Finding |
|---|---|---|---|
| S1 | **MED (boundary)** | train_probe.py:82; same pattern tools/inventory.py `_probe_one` (`[sys.executable, "-I", "-c", _CHILD, path]`) | The full real clip path, including the file name, is passed as an argv argument to the child. It shows up in `ps -axo args` / `pgrep -f` for the whole extraction time; I reproduced this with synthetic paths. Agents in this session do run `ps`. Fix: pass the path over stdin or a pipe, not argv. |
| S2 | MED (interpretation) | docs/COORDINATION.md:214, the stage-2 row | "A1 … no feature above null95 (best person_area_max 1.0 vs 0.81)" contradicts itself, since 1.0 > 0.81. Correct reading: one feature exceeds its own per-feature null95. With 22 features and n=5 (smallest attainable one-sided p = 1/32), about 1.4 features would hit the 1.0 ceiling by chance, so it is not significant, but A1 also cannot show absence. The planted perfect signal reached the same pAUC of 1.0. A1 has no family-wise (max-over-22) null, unlike A2 (:317-320). "Optimistic exploratory F1 0.615 … below noise floor" compares an F1 with an AUC floor, which is not valid; there is no permutation floor for F1. |
| S3 | MED (could hide signal) | train_probe.py:288-290, :331 | The label-time window is a symmetric ±5 / ±15 s around the imported timestamp, with no lag scan. The probe also never checks label times against clip duration as an aggregate. If imported times mark start or end, are approximate, or the auto-ru unit mapping is off, A1, `peak±5s` and the event metrics are misaligned while the clip-level AUC is not. Diagnostic to add: report the share of label times ≤ duration (aggregate). |
| S4 | MED (could hide signal) | train_probe.py:157 (`estimate_zone`) | The zone is the densest region of person-box centres; 70.6 % of samples have 2+ persons. If the zone locks onto a stationary client or a second employee, the zone and ring features measure the wrong role and an absence signal disappears. Nothing has been checked against a human-marked operator position; the planned attribute annotation should include that. |
| S5 | LOW-MED (could hide signal) | detector.py:250-253, train_probe.py:199-205 | Objects use conf 0.25 plus argmax-class gating on a COCO side-view model with top-view footage, so the object features may be almost always zero. The probe does not report per-feature non-zero rates, so "no separation" and "feature never fires" look the same. Add an aggregate firing rate per feature. |
| S6 | LOW-MED | train_probe.py:245-251 | The clip score is a sliding-window max. Its null value grows with clip length, and samples per clip range 74 / 102 / 611, which is large heterogeneity. Length is a confound in either direction. Diagnostic: report the AUC of clip length alone against the label, as an aggregate. |
| S7 | LOW | train_probe.py:136, :138 | The `assert side=="train"` is a tautology inside a loop over train shas, and `python -O` removes it. The sha re-check runs after the file is opened. With a corrupted index→sha mapping, a holdout file **is opened and fully read for hashing** before the refusal; an open() spy showed `c_hold3.mp4` being opened. Its content is never analysed. Better: an index→sha table verified without opening holdout files. |
| S8 | LOW | train_probe.py:117 | `load_imported` parses the whole label file, holdout statuses and events included, into memory. They are unused afterwards, so "never reads holdout labels" is not literally true. |
| S9 | LOW | train_probe.py:265, :356, :366 | `metrics()` gets no `n_failed`, so `extraction_failed_excluded` is always 0. The real figure is in `coverage.extraction_failed` (0 in the real run). The field misleads whenever failures occur. |
| S10 | LOW | train_probe.py:115-116 | `--candidate` is not validated: it goes straight into `os.path.join` (local CLI only). |

### Verdict on the negative conclusion
**Sound in its narrow, like-with-like form:** "With these 22 features / 176 configurations on 17 train clips (5
positive), there is no convincing separation. The best train AUC, 0.750, is below the permutation max-AUC p50 of 0.833
(p95 0.950) under the same best-of-176 selection. Nested clip F1, 0.3077, is below always-positive, 0.4545." No
leakage was found. Production stays fail-closed, which is correct.

**It is not evidence that no observable signal exists.**
- With n = 5 positives, the procedure can confirm only near-perfect separators: beating p95 0.95 needs almost perfect
  ranking.
- At n = 17, nested F1 swings around the baseline by chance: 0.50 on synthetic no-signal data.
- A1 cannot reach significance after correcting for 22 features.
- S3–S6 are concrete ways a real signal could be hidden:
  - label-time semantics and window placement
  - a zone on the wrong person
  - object features that never fire
  - clip-length confounding
- 16 of the 22 positives are in the frozen holdout.

The human attribute-annotation step, together with checks of the label times and of the operator zone, is the right
next action. It is not grounds for new features now.

## Stage 4: independent critical review of the nested preceding-window probe (critical-reviewer, synthetic only)
Scope: docs/PROTOCOL_STAGE4.md (incl. §3 pre-run amendment), tools/stage4_probe.py, tools/stage4_selftest.py,
tools/train_probe.py (features/metrics), tools/local_safe.py, tools/evaluate.py matcher. No video, frame, label or
human-answer content was opened. The real summary was inspected for **structure only** (205 leaves, longest list = 3,
no string values except the protocol tag, no "/" and no 16-hex sha-like strings). Audit hashes (attrs, R1, map
prefix): **unchanged**. Protocol sha256-16 is 673a3c005c8ca67a, the same as at run time. The real cache dir holds 17
files, so no holdout caches are present.

### Verified OK
- **Nested correctness (stage4_probe.py:103-121).** For outer clip k, `tr` excludes k (:110). Inner clip-LOO fits θ on
  15 clips (`it = tr[tr != j]`, :113-114) and predicts j. Inner F1 (:116) and the inner hit rate (:117) come only from
  `tr`. AUC is computed on `tr` (:118). The lexsort keys (F1, hit, AUC, lowest index) match §3. θ is refit on all 16
  (:119), and k gets one prediction (:120). `fit_thr` (:76-82) matches `train_probe.fit_threshold`: candidates are
  the training scores, and ties go to the higher threshold.
- **Leakage, by experiment.** Over 88 perturbations of the outer clip's own label and/or label time, the outer
  prediction, t_pred and chosen config changed **0 times**. This is not a power-less test: flipping one *inner* label
  changed another clip's prediction or config in 17/17 cases. `scan()` (:50-72) has the signature
  (t, x, f, W, sign, agg, off). It takes no label, label time, R1/T1/V1 or trim hint. Scan points cover the whole clip.
  `f` = samples/duration is observable at inference.
- **Permutation control (:143-148).** Each replicate shuffles labels, carries relative label positions over, and
  reruns the **full** `nested()`, inner LOO included. p_perm = (1+#≥obs)/(1+n).
- **Duration-only baseline (:340-343).** It runs through the same `nested()`. Its hit matrix is all zeros, so the
  selection key reduces to (F1, AUC, index), as §5 says.
- **Metrics.** Clip F1 for the decision and the null uses `clip_f1` (:124-127), which returns 0 when TP=0. Event
  metrics use `train_probe.metrics` → `evaluate.match_events` at ±5 s. There is one event per predicted-violation
  clip, every label time counts toward FN, and the hit rate is verdict-independent and labelled
  "timestamp-endpoint".
- **Isolation.** `select_split_units(..., "train", dry_run=True)` is used. Holdout records are discarded at parse
  time. The index-conflict and side checks are explicit `if` statements, not asserts. Cache paths are built only for
  train shas. Self-test (`LOCAL_PRIVATE_DIR` in scratch): **3/3 PASS**. The open() spy saw no holdout path, holdout
  files stayed mode 000, stderr was empty, and summaries contained no sha or path (my own grep agreed: 0 sha hits,
  0 "/").
- **Human files.** R1 is read only by `zone_groups` (:269-291), which feeds the ablation only. attrs is only hashed.

### Findings
| ID | Sev | Where | Finding |
|---|---|---|---|
| S4-1 | MED (could hide signal; scope of NO-GO) | stage4_probe.py:118 | Selection is **F1-first**; the inner hit rate only breaks ties. The hit criteria therefore test the localization of the *F1-selected* config, not the best localization the family can reach. Synthetic test (cue at the label in all positives, the same cue at random times in 50 % of negatives; 30 reps): median hit 0.8, but GO 1/30. |
| S4-2 | MED (could hide signal) | stage4_probe.py:71 (earliest argmax), §1 grid | Earliest-argmax plus W ≤ 15 s, offset ∈ {0, −2.5}: the scanner can only localize cue *ends* for cues ≲ 22.5 s long. With max aggregation it localizes the cue *onset*. Synthetic 30 s cue ending at the label: F1 1.0, hit 0.0, GO 0/30, while the best single config reached 0.8. |
| S4-3 | MED (power) | protocol §7 | With 5 positives and median duration ≈ 24 s, the permutation hit p95 is high (real 0.6; synthetic median 0.8). True positives keep their cue under permutation, and ±5 s covers ~40 % of a clip. Synthetic GO rate: 0/30 null, 93 % for a cue visible in 5/5, **33 % for 4/5, 13 % for 3/5, 0 % for 1/5**. The family (224 configs) gives a chance F1 p95 ≈ 0.73–0.75. A NO-GO is expected unless the cue is near-perfect. |
| S4-4 | LOW (would make a GO optimistic, not the NO-GO) | §7 | The GO rule does not require beating the duration-only baseline (0.600) and does not use the |ρ|≥0.5 duration flag. Random-time is uniform over [0,d], not over the scanner's valid τ support. "F1 > always-positive" is passed by chance in ~33 % of null runs. |
| S4-5 | LOW (reporting) | evaluate.py:171 (old) | `_prf` returned F1=None for TP=0 with FP+FN>0 (reproduced on the synthetic nosignal fixture: TP0/FP3/FN5 → None). It is now fixed to 2TP/(2TP+FP+FN). The decision, the null and the primary numbers are unaffected: primary clip TP=3 and event TP=1; the decision uses `clip_f1`. Only the display of the ablation event F1 changes (→ 0.0). |
| S4-6 | LOW | stage4_probe.py:180 | Inside `random_time`, the interval-union step lacks `max(0, …)` for non-final intervals. Two or more label times beyond the duration in one clip give negative coverage. This cannot occur in the real run (5 label times for 5 positives). |
| S4-7 | LOW | stage4_selftest.py:7 vs :33 | The docstring says the "missing" scenario is *planted*, but `plant()` plants only when scenario == "planted", so "missing" is a no-signal fixture. The decode-path test is still valid; only the description is wrong. The positive control (60–150 s clips, exact 5 s cue) is much easier than the real data (median 23.8 s). |
| S4-8 | INFO | stage4_probe.py:89 | numpy prints a RuntimeWarning (−inf − −inf) to stderr when a clip has no valid window. It carries only the repo source path, no data. |

### Verdict
- **The protocol is leakage-free and the nesting is correct.** This was verified by line-by-line reading and by the
  perturbation experiment. The duration baseline and every permutation run the same nested scheme.
- **The measured NO-GO is methodologically sound and robust.** All four GO criteria fail, and none of them
  narrowly:
  - F1 0.400 < always-positive 0.4545, and < perm p95 0.727;
  - hit 0.200 < random-time 0.387, and < perm p95 0.600.
  - No defect found (S4-5, S4-6) changes any criterion. The result is not optimistic.
- **Scope of the conclusion.** The NO-GO applies to *this candidate family and selection procedure on this small
  train sample* (n = 17, 5 positives). It is not evidence that no observable signal exists. Per S4-1..S4-3, the
  procedure has low power except for near-perfect cues, and it cannot localize long events.
- **Interpretation.** The audit aggregates (V1 4 unseen / 1 partial; T1 4 end / 1 middle) constrain interpretation.
  They do not establish the cause of the failure. No causal claim ("label semantics, not features") is supported.
- **Disclosure.** §8 discloses the audit-informed design on the same train sample, and that is adequate. Not
  disclosed in §8: the holdout was already seen once in stage 1. Any future holdout use is therefore not fully
  virgin, and that should be stated wherever holdout numbers appear.
- **Holdout: NOT approved.** The NO-GO stands, so no locked holdout evaluation is justified. Production stays
  fail-closed.
- **Post-hoc rescue is not permitted.** Changing the family or the selection after seeing these numbers (for example,
  hit-first selection, latest-argmax, narrower families) would be post-hoc on the same train sample and would need a
  new stage with new justification.

## Stage 5: independent critical review of Part B (critical-reviewer, claude-opus-5-5, 2026-10-18, synthetic only)
Scope: tools/stage5_probe.py, stage5_model.py, stage5_encoder.py (embed child, watchdog), stage5_selftest.py,
local_safe.py against docs/PROTOCOL_STAGE5.md. The real run was not repeated and the holdout was not touched. Only
the aggregates and structure of `stage5_summary.json` were read.
- **Hashes.**
  - PROTOCOL_STAGE5.md sha256-16 recomputed = **b0d73d7ab46da775**. It matches the lock and the summary, and no
    amendment is appended.
  - The immutable files are all **unchanged**: attrs ab7a86b1…896f, R1 09247337…ac35, map e413108f8b7f0080 and
    stage4_summary 284e70c08e70c02b.
  - vision.onnx = 4e95b080c23eff44 (pinned).
- **Summary structure.** The summary has no per-clip rows, shas, paths, Cyrillic text or prompt text. The only
  hex token that contains letters is the protocol hash; the other 39 "hex" hits are float digits.
  - Coverage: 17/17 ok, 0 caches present at the start, 0 zone fallbacks, child peak 1110 MB, parent 392 MB.
  - Runtime: 718.8 s; n_perm 1000/1000.

### Findings
| # | Severity | Where | Finding / reproduction |
|---|---|---|---|
| C5-1 | MED (validity; could HIDE a signal) | stage5_model.py:105-115, :71-76 | **Scene identity.** Positive step targets are the [label−10, +2] windows of 5 clips, and the negatives are *all* steps of 12 other clips. Within-positive-clip contrast is unused, by design. With PCA-8 fitted on all steps, the leading components mostly carry between-clip scene variance. The model can therefore learn clip/scene identity rather than an action, and that fails under clip-LOO. The residual features only partly offset this. Consistent with the outcome: 9 distinct configs across 17 folds, and F1 0.30 below the null median 0.375. This is a design limit, not leakage. |
| C5-2 | MED (power) | protocol §7 vs real perm null | **The GO bar is very high.** With 5 positives, the perm hit p95 is 0.8 and the rule is strict `>`, so a GO requires **5/5 hits**. F1 must exceed the perm p95 of 0.667. Numpy plant experiment (real `M.nested`, 4 reps per level, no perms): cue 0.1 → hit 0.35, P(F1>0.667) 0; cue 0.2 → P(hit = 1) 0.75, P(F1>0.667) 0.25; cue 0.3 → 1.00 / 0.75. **The F1 criteria bind first.** The NO-GO is evidence only against strong, consistent cues; det-stage5's §9 power check quantifies this. |
| C5-3 | LOW (threshold calibration; depresses F1, not leakage) | stage5_model.py:201-202 | **Positive-heavy outer verdicts.** θ\* is an F1-max threshold over 15-clip inner held-out scores, applied to a 16-clip outer score. Null synthetic (6 reps): 72.5 % of outer verdicts are positive (0.59–0.82) and F1 mean is 0.29, below always-positive 0.4545. Real run: 15/17 predicted positive (TN = 0). The permutation null reproduces the effect, so p-values stay calibrated. Criterion 1 is reachable only with a strong signal. |
| C5-4 | LOW (would make a GO optimistic; NO-GO unaffected) | stage5_model.py:212-219, protocol §6 | **Weak random-time baseline.** It is uniform over the clip or the valid support. With T1 aggregate 4 end / 1 middle, a trivial label-free "clip end" localiser could approach 0.8. Criterion 5 covers this, because the permutation keeps relative label positions (perm hit p50 0.4, p95 0.8). Criterion 4 alone would not. |
| C5-5 | LOW (fail-closed completeness) | stage5_probe.py:129-131, :201 | **Selector skips are not failures.** `primary_complete` = (failed == 0). A unit skipped by `select_split_units` (status conflict, bad index) is not counted as failed, so 16 units could still yield "complete". Fix: also require label_units_selected == 17 and skipped_by_selector == 0. Real run: 17 selected, 0 skipped, so it is unaffected. |
| C5-6 | LOW (reproducibility) | stage5_probe.py:80, :108-125; stage5_encoder.py:314-320 | **Cache trust.** Existing embedding caches are reused without checking the ONNX hash, decode completeness or zone. `text_probes.npy` has no pin; its sha16 is e5c2ba8d958bfc2a, created 20:24 UTC, before the 20:29 lock. Real run: 0 caches at the start, so it is unaffected. Re-runs should pin both. |
| C5-7 | LOW | stage5_model.py:131-135; S4.fit_thr | **No-valid-step clips.** Protocol §3 says such a clip is "excluded and counted". In code it gets score −inf and is not counted. A −inf training score can become an F1-max threshold candidate, which makes every clip positive. Not triggered: min 17 frames, 622/656 valid steps (= 656 − 2×17). |
| C5-8 | LOW (reporting) | stage5_probe.py:255-265 | **Partial sensitivity/ablation runs.** If their permutations break early on the deadline, they are not added to `incomplete_phases`; only `perm.complete` shows it. Real run: 200/200 for both. |
| C5-9 | INFO (privacy hygiene / disclosure) | stage5_probe.py:278, :276-277 | `stage5_summary.json` is written 0644 (the private inputs are 0600). Coverage min/med/max, which the protocol allows, exposes one clip's duration (the max). The machine-readable disclosure string omits "holdout seen once in stage 1" and "audit-informed design"; the protocol states both. |
| C5-10 | INFO (checked, not supported) | stage5_model.py:131-140 | **No length bias from the max-logit score.** I suspected that a max-over-steps clip score would favour the long negatives. Null synthetic, durations 17–142 s, 3 reps: signed Spearman(outer score, duration) per fit-config was between −0.09 and +0.18. No material bias. Real chosen-score median \|ρ\| = 0.33. |

### Verified OK (line reading + synthetic experiments)
- **Leakage: none found.**
  - The scan is label-free: `prep` uses the whole-clip median, `localise` covers all valid steps, and the zone comes
    from `TP.features` (persons → `estimate_zone`).
  - k's label, time, sha and R1 are never inputs to k's prediction. Sin[g,k,j] is trained on idx∖{k,j} and
    Sout[g,k] on idx∖{k} (stage5_model.py:152-166). Targets are used only for train clips (:120-121). PCA and
    standardisation are fitted on train steps only (:71-83).
  - Thresholds and selection use only Sin[·,k,tr] (:189-201), and there is one outer prediction per clip.
  - R1 is used only in the ablation (stage5_probe.py:254-263), after the primary analysis.
  - **Perturbation experiment** (scratch, real `M.nested`): 34 outer label flips and time shifts gave **0** changes
    to k's own prediction, time or config. A deliberately leaky variant, where the outer model includes k, gave
    **17/34** changes, so the test detects leakage.
- **Permutations.**
  - Each of the 1000 replicates reruns the entire §5 procedure: all fits, inner thresholds, J selection, refit and
    the outer threshold. Only the label-free fold objects are cached.
  - Labels are shuffled with relative positions rescaled (stage5_model.py:222-226), and the seed is fixed.
  - n_perm comes from the CLI and the loop only breaks on the deadline (stage5_probe.py:151-153). It was 1000/1000,
    as pre-registered.
  - The duration-only baseline is nested like stage 4 (S4.nested: inner LOO threshold, sign selection, refit).
  - Random-time is the max of the uniform-clip and valid-support baselines.
- **Guards.**
  - There is one deadline across embedding (<30 s left → `incomplete_budget`, counted as failed), the primary
    permutations, sensitivity and ablation.
  - The decode check is fail-closed (stage5_encoder.py:274-278). Failures write no cache (stage5_probe.py:102-107),
    the unit counts as failed, and `decide` blocks GO when `primary_complete` is false.
  - Criterion 6 treats NaN as 1.0, i.e. fail.
  - The zone policy is single (fallback is counted, never excluded).
  - **Memory watchdog re-run by me**: `memtest --mem-mb 256 --alloc-mb 300` → status=memory_cap, peak 271 MB, 0
    leftovers. Control with cap 1024 → status ok, peak 330 MB.
- **Metrics.**
  - Clip P/R/F1 come from the outer verdicts, and events come from `match_events` ±5 s.
  - `_prf` is fixed: event TP0/FP15/FN5 → F1 0.0.
  - The hit rate is labelled as a timestamp endpoint.
  - All 7 criteria in `decide` match §7.
- **Privacy.** Child jobs go over stdin with stderr to DEVNULL. Only numeric JSON comes back. The cache is under
  local_private (gitignored), and there is no prompt text in the outputs.

### Verdict
- **Leakage-free: YES.** No path from an evaluated clip's label, time, filename, hash or human R1/T1/V1 into its own
  scan, selection, threshold or prediction. This was proven by the perturbation test plus the leaky-variant control.
- **The NO-GO is methodologically sound and robust, not borderline.** 5 of 7 criteria fail by wide margins:
  - F1 0.30 vs always-positive 0.4545, duration-only 0.60 and perm p95 0.667 (p_perm 0.77);
  - hit 0.20 vs random-time max 0.424 and perm p95 0.8 (p_perm 0.89);
  - event F1 0.0.
  - The observed results sit below the null medians. No finding above changes any criterion. The sensitivity
    (full-view) and ablation runs fail too; they are not used for the decision.
- **Scope.** The NO-GO is for this encoder, these features and this selection, on n = 17 / 5 positives. Power is low
  except for strong, consistent cues (C5-1, C5-2). It is not evidence that the behaviour is invisible.
  - The audit aggregates (V1 4 unseen / 1 partial; T1 4 end / 1 middle) constrain interpretation only. No causal
    claim is supported.
- **Disclosure: adequate in the protocol.** It covers the adaptive exploratory work on the same train sample after
  stages 2–4, the absence of independent validation, the holdout seen once in stage 1 and now closed, and the
  inner-selection optimism. The optimism affects selection only and is reproduced in the null, so it cannot inflate
  the outer metrics. Add the stage-1 holdout note to the summary string (C5-9).
- **Holdout evaluation: NOT approved.** There is no GO, and production stays fail-closed.
- **Post-hoc rescue is not permitted.** Examples: within-clip negatives, other thresholds, other families, or a
  clip-end localiser on the same sample. Any of these would need a new stage.

## Stage 7 — pre-lock (critical-reviewer, claude-opus-5-5, 2026-10-18, synthetic only, no real data)
Scope: PROTOCOL_STAGE7.md as read, sha256-16 **80a035fec17546f6**. Lock candidate re-checked: **7ebbd0602bedba0f**.
Also tools/stage7_core.py, tools/stage7_wcr.py and tools/stage7_selftest.py. All experiments ran on synthetic fixtures under the
scratchpad `prelock7` (LOCAL_PRIVATE_DIR there). No video, frame, label or real cache was opened.

### Verified OK
- **No circular-shift null.** No `roll`, circular or cyclic code exists in the stage7_* files.
- **N1 is non-degenerate.** On one AR(0.9) series, u had 58 distinct values over 200 transplant positions (sd 0.287).
  Over cyclic shifts, the max had 1 distinct value and u at a fixed position had 57.
- **N1 boundaries.** τ snaps to the nearest valid step. The end anchor clamps at 0.
- **Clip-wise partition.** No scored clip is ever in the set that fits its scale, direction or threshold. This holds
  in the main run, every N2 replicate and N1 (core.py:78-84, 100-106, 164-181, 201-231). The only exception is the
  per-clip median centring, which is label-free.
- **Direction.** It is fixed, with no grid, regularised by a robust diagonal scale and a unit norm.
- **Multiplicity and GO bars.** Bonferroni is named and applied correctly (core.py:14). GO is a superset of the
  stage-5 seven, with stricter p bars, and adds criteria 8, 9 and 10 plus duration-only (criterion 2). No bar was
  lowered.
- **Watchdog, privacy and immutables.** The memory watchdog kills at the cap. Outputs carry aggregates only, and
  the summary file is mode 0600. Immutables are hash-compared.
- **Self-test.** `--n-perm 100 --power-reps 5` passed 10/10 in 59 s wall, max RSS 164 MB. Power: amp 0.12 5/5 → GO
  2/5; amp 0.15 5/5 → 1/5; amp 0.15 4/5 → 1/5.
- **False GO is 0/60** in each of three null fixtures (n_perm 100): plain null, heteroskedastic positives
  (noise 1.6×), and end-anchored labels with a common end transient.
- **R1–R6 are applied faithfully in 7ebbd060.**
  - §3: the exact validity conditions of N2 and N1 are stated, the guarantee is removed, the residue is marked
    UNKNOWN / UNVERIFIED, and "UNVERIFIED controls limit interpretation even under a numeric GO" is stated.
  - §6: the anchors' edge-distance limits and the first-label-only rule are stated.
  - §7: criteria 5 and 8 are declared not to be localisation tests, and the meaning of a GO is stated.
  - §11: power is given as counts with Clopper–Pearson CIs, plus the replicated-null false-GO rate of 0/60.
  - §3: the signed ρ and the negatives-only ρ are added as diagnostics.
- **The code diff is output-only.**
  - `rho_neg` (core.py:163, 179-181) and `DIAGNOSTIC_signed_rho_…` (core.py:316-318, wcr.py:280-282) feed no
    criterion, threshold or selection.
  - Criterion 6 still reads only `r["rho"]` (core.py:302, 306).

### Findings
- **S7-1 (MEDIUM; new, found after R1–R6). N1 is anti-conservative by roughly 2× at its nominal level.**
  - Location: core.py:222 (and protocol §6 "Null draw"). The null resamples only the 12 *realised* negatives, so it
    ignores the sampling error of that reference set.
  - Expected effect: the variance ratio is (1/5 + 1/12)/(1/5) ≈ 1.42. A nominal 0.025 then gives about 0.05, and a
    nominal 0.05 about 0.08.
  - Measured, both anchors at 0.025, plain synthetic null:
    - 3/60, 2/60, 12/200 and 6/200, pooled 23/520 = 0.044 (binomial p ≈ 0.004 against 0.025);
    - at 0.05: 0.055–0.09.
  - Equal durations did not remove it (0.05 over 150 seeds), so the cause is not duration.
  - Fix, stricter and lowering no bar: include the observed u_k in each stratum's reference, i.e.
    `pool_in = np.c_[pool, q]` and J_k uniform over the 13 values. Measured after the fix: 4/200 = 0.020 at 0.025,
    and 10/200 = 0.050 at 0.05.
  - The overall false GO stays 0/60 under the intersection, but criterion 9 is the stated localisation evidence.
- **S7-2 (LOW, disclosure).** Criterion 6 under the null passed 41/60 (median |ρ| 0.46, close to the bar). On real
  data, criterion 6 can decide a NO-GO by length alone. This is covered by the new diagnostics.
- **S7-3 (LOW).** The real position-prior hit is UNKNOWN before the run. If the labels sit near the end, criteria 5
  and 10 may be structurally unpassable (§11 `nearend`). This is already disclosed.

### Verdict
- **REVISE, one item (S7-1).** Apply the observed-included N1 reference as a dated pre-run amendment: §6 text plus
  core.py:222. Then re-run the self-test; it must pass 10/10.
- After that, **APPROVE FOR LOCK** without further review beyond a diff check.
- If the coordinator declines the code change, the minimum is an amendment that states N1's measured synthetic
  level (≈ 0.044 at nominal 0.025). Criterion 9 is then interpreted at that level, and I would approve that version
  too, with the defect recorded.
- A technical PASS or a numeric GO remains exploratory and is not detection evidence.
- **Post-fix diff check (lock ab2c71c974fa9d7d, 403 l): APPROVE FOR LOCK.**
  - transplant() core.py:225-227 matches the fix I tested: `pool_in = np.c_[pool, q]`, with 13-value draws. Nothing
    else in the numeric path changed.
  - §6 "Null draw" (l.236-240) describes it exactly.
  - S7-1 is resolved.
  - Caveat: "0/60 is an upper bound" holds stochastically. The new p is not ≥ the old p draw by draw, because the
    random draws differ.

## Stage 7 — post-run (critical-reviewer, claude-opus-5-5, 2026-10-18 ~02:50 UTC; read-only; no real re-run)
Scope: COORDINATION "STAGE 7 REAL RUN DONE", PROTOCOL_STAGE7.md, tools/stage7_core.py, tools/stage7_wcr.py, and
`stage7_summary.json` (structure and aggregate fields only). Synthetic checks ran in the scratchpad `post7`
(LOCAL_PRIVATE_DIR there). No video, frame, label row or real cache was opened.

### Hashes (recomputed)
- PROTOCOL_STAGE7.md sha256-16 **ab2c71c974fa9d7d** (mode 444), which equals `protocol_sha256_16` in the summary.
- Immutables unchanged:
  - attrs ab7a86b1…f4896f and R1 09247337…7c97ac35: full 64-hex match;
  - map e413108f8b7f0080; stage4 284e70c08e70c02b; stage5 5d399b48ca612b9e;
  - the summary reports `immutable_files` n 6 `unchanged`.
- Code: core b11ed711d908be33, wcr 2fda52ab0c8070bf, selftest 8c3050fce10595be. The code mtimes (02:33:58 and
  02:26:23 UTC) and the protocol mtime (02:35:15) all precede the summary (02:37:29).

### (1) Conformance: YES
- n_perm 1000 with 1000 completed in both families; seeds come from code, SEED + family index.
- Both families were run, 17/17 units, 0 failed or fallback.
- N1 uses the 13-value pool (core.py:225-227) and is non-degenerate: pool distinct 48–53 and null distinct ≥ 9569
  for every anchor.
- N2 is complete.
- All baselines are present: always-positive, duration-only nested, random-time (both), and both position priors.
- Re-run or re-selection: no evidence of either.
  - The summary is written with O_TRUNC (wcr.py:248), so an overwrite would keep the original birth time. Here
    birth equals mtime (02:37:29), so the file was written once.
  - A deleted earlier file cannot be excluded from metadata alone.
- **P7-1 (LOW, process).** The summary was written at 02:37:29 UTC. That is about 2.5 min before the logged
  "LOCKED (chmod 444) at 02:40 UTC … ONE real run authorised now". The run sha matches the sha that was later
  approved, and no file changed after the run, so validity is unaffected. The coordinator should reconcile the log
  time.

### (2) Criteria table: CORRECT
I recomputed every criterion from the aggregates. Bars are at α = 0.025.

Semantic family (A_sem):

| # | Value | Bar | Result |
|---|---|---|---|
| 1 | F1 0.300 = 6/20 | > always-positive 0.4545 | fail |
| 2 | F1 0.300 | > duration-only 0.600 | fail |
| 3 | p 0.841 | ≤ 0.025 | fail |
| 4 | hit 0.80 | > max(0.387, 0.424) | **pass** |
| 5 | p 0.089 | ≤ 0.025 | fail |
| 6 | median \|ρ\| 0.215 | < 0.5 | **pass** |
| 7 | 17/17 | complete | **pass** |
| 8 | event F1 0.200 = 4/20, p 0.344 | ≤ 0.025 | fail |
| 9 | p 0.0141 / 0.0034 | both ≤ 0.025 | **pass** |
| 10 | MAE 3.10 | < min(2.73, 6.04) | fail |

Motion family (A_mot):
- Passes: 4 (hit 0.60 > 0.429), 6 (\|ρ\| 0.059) and 7.
- Fails: 1 and 2 (F1 0.267 = 4/15), 3 (p 0.867), 5 (p 0.389), 8 (p 0.189), 9 (p 0.164 / 0.097) and 10 (MAE 4.30).

Confusion counts sum to 17 in both families. The `event_f1_matches_evaluate` cross-check is true for both.
NO-GO follows by rule.

**P7-2 (LOW, reporting).** In the COORDINATION semantic line, "observed rank 0.811/0.662" reads as two semantic
anchors. In fact Q_observed is 0.811 for both semantic anchors, and 0.662 is the *motion* Q. This should be corrected
in the final report.

### (3) The semantic N1 pass
"Not evidence of detection" is the correct reading. The reasons:
- N1's validity condition (§3/§6) is unverified.
- The pass is one criterion of a failed 10-criterion intersection test, and it is reported after the result was
  seen.
- The motion family does not reproduce it.
- It co-occurs with a failure of criterion 10, which fits a broad late-clip difference better than a sharp event
  peak.

A concrete non-event mechanism exists:
- In short positives, B is mostly the clip start (t < tl − 15), so w ≈ "label window minus clip start of positive
  clips".
- Any shared positive-only cutting structure therefore raises u_k.
- **Synthetic check** (`post7/cut_alt.py`, 40 seeds each, near-end labels, no event at the label):
  - null: c9 pass 0.025;
  - positive-only clip-start state, amplitude 0.15 / 0.30: c9 pass **0.125 / 0.150**, c10 pass 0.025 / 0.05;
  - positive-only clip-end state: 0.05 / 0.10.
  - So the observed pattern (c9 pass, c10 fail) arises without any event. This result is specific to the synthetic
    model and is not a claim about the real cause.

The pass may be reported as an **exploratory observation** with this wording:

> "Exploratory observation, not a detection result: in the SigLIP2 zone family, the out-of-fold score at the labelled
> second ranked higher within its own clip (mean percentile 0.81, 5 positives) than at position-matched seconds of
> the 12 negative clips (transplant null p = 0.014 relative / 0.003 end-anchored; per-family bar 0.025). The family
> failed 6 of 10 pre-registered criteria, including clip F1 (0.30 vs always-positive 0.45 and duration-only 0.60) and
> localisation (MAE 3.10 s vs 2.73 s for a relative position prior with equal hit 0.80). The null's validity condition
> is unverified. The difference is equally consistent with how positive clips were cut, where annotators placed the
> endpoint, or a non-violation scene change near the label. The motion family did not show it (p 0.16 / 0.10). This
> was the third adaptive look at the same 17 train clips, with no independent validation."

### (4) Localisation: not identifiable here (CONFIRMED)
- The relative prior reaches MAE 2.73 s with hit 0.80. The model reaches 3.10 s with hit 0.80.
- The prior alone passes criterion 4, so the criterion 4 pass is uninformative.
- N2 p(hit) is 0.089.

### (5) Pre-check η²: partly overreaching
- "Partly supports scene identity" is acceptable only with a chance reference.
  - Synthetic (`post7/eta_pca.py`, 20 seeds): with **no** scene offset (AR drift only), PCA-8 η² already has a
    median of 51–65 %; with a scene offset it is 99.7 %.
  - The real E_zone medians (68 %) are therefore weak support. The maxima (97 % / 95 %) are the stronger indication
    that some components are clip-specific.
- **P7-3 (LOW-MEDIUM, wording).** "Per-clip scale survives centring (20–27 %)" is not supported.
  - On a homoskedastic synthetic fixture (identical scale in every clip), the centred-step-norm η² is 33–48 %
    (`post7/eta_null.py`, 40 seeds: median 48.0, p95 53.6). The i.i.d. reference is 16/621 = 2.6 %.
  - High-dimensional norms concentrate, so between-clip drift dominates η².
  - Suggested wording: "centred-norm η² 20–27 %; no valid chance reference; uninterpretable as evidence for or
    against residual per-clip scale."

### (6) Duration and TN = 0: consistent with C5-3; S7-2 did not materialise
- Duration-only F1 is 0.600, and all 17 folds chose shorter = violation. It remains the strongest baseline.
- In the semantic family, 15/17 clips are predicted positive with TN 0. Observed F1 (0.30) is below the N2 median
  (0.38). This is F1-max threshold collapse without separation (C5-3); the midpoint threshold does not prevent it.
- The signed ρ is +0.215 (longer clips score higher) and the negatives-only ρ is +0.063. This fits the designed
  length bias that favours the long negatives.
- Criterion 6 passed in both families, so it did not decide the NO-GO (the S7-2 risk was not realised).
- A passed criterion 6 does not show that there is no length effect: the maximum \|ρ\| is 0.47.

### (7) Privacy
- The summary carries aggregates only. A scan found no shas or paths, and there are no per-clip rows or times.
- **P7-4 (LOW).** The summary mode was changed from 0600 to 0444 (world-readable), and `local_private/` and
  `train_probe/` are 0755. Content is aggregate-only; suggest 0400 and a 0700 directory.

### (8) Honest conclusion for the final report
- **Established:**
  - The single locked attempt ran as pre-registered.
  - Both families are a measured NO-GO.
  - Clip-level output is no better than always-positive or duration-only.
  - Localisation is no better than a position prior.
  - Production stays fail-closed.
- **Plausible:**
  - In SigLIP2 zone features there is a positive-clip-specific, label-co-located within-clip difference of unknown
    origin. Clip cutting, endpoint choice and non-violation change are not excluded.
  - Clip identity carries substantial embedding variance.
- **Unknown:**
  - whether any visual correlate of a violation exists;
  - whether the labels are visually verifiable (V1: 4/5 unseen);
  - performance on any independent data.
- **Required caveats:**
  - adaptive exploratory research on the same train-17 (5 positives): third look by protocol count, fourth counting
    the human audit;
  - no independent validation;
  - the holdout was seen once in stage 1 and stays closed;
  - low power (synthetic 3/5 at a strong 5/5 cue, 0 with near-end labels);
  - the validity conditions of N1 and N2 are unverified;
  - absence of signal is not shown.

### Verdict
- Run conformant: **YES**, with P7-1 (timestamp) to reconcile.
- NO-GO sound: **YES**. It is robust even if N1 is taken as valid, because criteria 1, 2, 3 and 10 fail.
- Holdout: **NOT APPROVED.** §7 licenses a holdout proposal only after a GO. Testing the N1 observation on the
  holdout would be a post-hoc hypothesis and would need a new pre-registration and a user decision.

## Stage 8b — pre-lock (critical-reviewer, claude-opus-5-5, 2026-10-18)
Scope: docs/PROTOCOL_STAGE8.md (sha256-16 ca8da70e2517d216, recomputed), tools/stage8_audit.py (b4f33f6d6bcda2c0),
tools/stage8_selftest.py (1005cd93eeac6cb1); lock digest recomputed f8c7c0c6d62c25f5 (matches arch report); snapshot
sha256-16 31c0a7548a928fd2 (matches pin); 8/8 immutables match recorded sha16; files still 0644; no stage8 summary
exists. Real labels/videos not opened. Self-test 4/4 PASS (stderr empty); pytest 172 passed.

### Verified
- Protocol wording: "not three independent clip units" (§3 l.79-80); one-sided sign-flip min p 1/2 (l.81);
  C(6,3)=20 / C(7,3)=35 only as hypothetical unpaired examples, no "smallest" (l.74-78); 4/3 = operational screen
  derived from ">=3 positives & >=2 controls per LOO fold", "not a universal necessary condition", not a validity
  certificate (l.66-73); stage 5/7 GO bars unchanged (l.26); U separate, never negatives (l.54-55); imported negatives
  excluded (l.56-57); descriptive audit considered and excluded (§6); disclosure 4+ reuse, no inference (l.17-20).
- Gate code: latest rev per task_id, same rule as read_log (stage8_audit.py:101-109 vs annotate_intervals.py:149-150);
  P requires both bounds non-None (121-122, 188-189), and clean_payload guarantees numbers; confident control = both
  confirmations + good + sure + numeric (131-133), external via classify == normal_confident, which clean_payload ties to
  both confirmations (annotate_intervals.py:117-118, 132); non-train records excluded and counted (145-147);
  thresholds hard-coded (37), pinned C02 snapshot not overridable (82-83), no env/CLI threshold path; outcome
  STOP_INSUFFICIENT_DATA + unmet numbers (201-205, 223-224).
- Holdout/refusals: selection via select_split_units(dry_run) discards other-side records at parse (local_safe.py:78-81);
  no clip/cache open (self-test open() spy); lock check -> summary-exists check -> sha check before parse (249-253,
  99-100); O_EXCL 0400 write once (263).
- Leaks: summary/stdout carry counts, enum keys, code-file sha16 and the lock digest only; the snapshot sha is reported
  only as a boolean; self-test leak check passes in all 3 scenarios.

### Findings
- **S8b-1 MEDIUM (methodology, outcome-invariant).** A confident same-clip control in a positive task flagged
  unknown/not_visible/event_unconfirmed is counted in C (stage8_audit.py:164-187: ivs = [] for k != "intervals", so
  no overlap check is possible, and the control still enters `ctls`). clean_payload allows clip_flags together with
  same_clip_controls (annotate_intervals.py:95-108). Synthetic repro: one not_visible task with one good/sure control
  gives C = 1, U = 1. The protocol says controls must not overlap an event, but an unlocalised event cannot be
  excluded, so such a control is not a verified normal. Edit: in gate(), count same-clip controls toward C only when
  k == "intervals"; otherwise add them to a new counter `n_same_clip_controls_in_unknown_tasks`. In §2 l.52 write
  "an entry in a done positive task of class intervals that does not overlap ..."; in §5 add the counter. Add one
  self-test case. The STOP outcome cannot change, since P = 2 < 4 decides it, but the reported C value could.
- **S8b-2 LOW (wording).** §3 l.84 says "passing the screen is necessary, never sufficient", which contradicts l.72
  ("not a universal necessary condition"). Edit: "passing the screen is required by this protocol before any further
  step; it is never sufficient."
- **S8b-3 LOW (theoretical).** P and external C count tasks, not distinct sha16 (synthetic: two control tasks on one
  negative clip give C = 2; two positive tasks on one clip give P = 2). The app cannot produce this: there is one
  positive task per clip and make_assignment gives disjoint candidates (annotate_intervals.py:178-179). Optional: count
  distinct sha16. Not required for the lock.

### Verdict
**REVISE (minor), then lock**: apply S8b-1 and S8b-2. S8b-3 is optional. After the edits: recompute the 3 sha16 values
and the lock digest, re-run the self-test (5/5 with the new case) and pytest 172. No re-review is needed beyond a diff
check of these edits. If the coordinator prefers not to touch code, the minimum acceptable change is the §2 + §3
wording plus a disclosure that same-clip controls in unknown-flagged clips are counted in C. That is weaker.

### Stage 8b — pre-lock diff check (after S8b-1/S8b-2)
- **New sha16 values (recomputed):** PROTOCOL 0a1778f3dbeaa34a, audit 4a7462f046bae17f, self-test 5a12b6edba65a8e8.
  Lock digest 5c94785af7be6620; thresholds still 4/3.
- **How I compared:** the files are untracked, so there was no textual diff. I compared against my own reading of
  ca8da70e/b4f33f6d. The self-test was checked by its new case and by running it.
- **S8b-1 applied in the strong form** (stage8_audit.py:176-191). A same-clip control enters C only in a task that
  counts toward P. Every other same-clip control goes to `n_same_clip_controls_in_unknown_tasks` (in the keys tuple,
  l.197). This is stricter than requested: an intervals task without a numeric interval is also excluded. That is
  conservative and acceptable.
- **Protocol updated to match:** §2 l.52-55, §5 l.124 and §8 l.194-200 ("4 scenarios", "5/5").
- **S8b-2 applied** (l.87-88).
- **Re-run:** self-test 5/5 PASS with stderr 0 bytes; pytest 172 passed. The synthetic probe now gives C = 0 for an
  unknown-task control.
- **No other change seen.**
- **Verdict: APPROVE FOR LOCK** (lock digest 5c94785af7be6620).

### Stage 8b — post-run conformity (critical-reviewer, claude-opus-5-5)
- **Integrity.**
  - Summary: 0400, 2284 B, sha256-16 b30752f0a5bb9397, matching the coordinator's report.
  - The 3 locked files are 0444. The lock digest recomputes to 5c94785af7be6620, with the same file sha16 values as
    approved.
  - Snapshot verified and unchanged; immutables 8/8, unchanged.
- **Structure: aggregates only.**
  - Integers, enum-count maps and fixed unmet strings.
  - No path separators.
  - The only 16+ hex strings are the lock digest and the 3 code-file sha16 values.
  - The only float is runtime_sec.
  - No ids, times or snapshot sha.
- **Outcome follows from the locked rules.**
  - P = 2 < 4.
  - C = 0 + 0 external = 0 < 3. The single same-clip entry lies in an unknown-flagged task, so it is counted in
    `n_same_clip_controls_in_unknown_tasks` = 1, as amended §2 requires.
  - STOP_INSUFFICIENT_DATA, with both thresholds unmet.
- **Internal consistency.**
  - 5 done = 2 interval tasks + 3 unknown; flag counts sum to 3.
  - 2 numeric intervals, 1 good+sure, 0 excluded non-train.
- **Note for docs.** §4's pre-run expectation said "1 same-clip confident control". The observed C = 0 comes from the
  S8b-1 rule locked before the run, not from any change in the data, and must be reported that way.
- **Verdict: CONFORMANT; STOP_INSUFFICIENT_DATA valid.** No inference, no training, holdout closed.

## Stage 9 — feasibility (critical-reviewer, claude-opus-5-5, independent; read-only; no data or caches touched)
**Verdict: NO_TRAINABLE_SUPERVISION. Stop now, and do not run or simulate tools/stage9_head.py.**

**Why the strict mask leaves nothing to train on.**
- Under the strict stage-8 mask there are 0 negative rows.
- The only confirmed control lies in an unknown-flagged task. That task has no event interval, so its non-overlap with
  the event cannot be verified.
- The stage9_head draft would therefore return its own `no_negative_supervision` / STOP_NOT_TRAINABLE (l.208-209).
- With positive targets only, the logistic loss is minimised by a constant head. Any such fit is meaningless.

**Rejected alternatives.**
- **Out-of-interval time of the 2 positive clips as negatives.** Not annotated as normal, so it is unknown. Forbidden.
- **Imported clip-level negatives, clip-level MIL (5 or 2 positive bags vs 12).** I do NOT justify their use:
  - this is the clip-level problem already run in stages 4/5/7 (NO-GO, train reused 4+ times), so the intervals add
    nothing to the negative side;
  - the imported negatives are not verified normal (stage-8 rule);
  - with 2 localised positives, leave-one-out leaves 1 positive per fold and the within-clip minimum p is 1/2, so no
    held-out unit exists and any number is resubstitution only.
- **One-class / positive-only density on about 2 intervals of frames.** Nothing to discriminate against and nothing
  to test. It would be uninformative about violations.

**What would unblock it (a data requirement, not a criterion change).**
- At least 3 confirmed-confident control clips, from external control tasks or from same-clip controls in interval
  tasks.
- At least 4 positives with numeric intervals, per the locked stage-8 screen.
- Then a new pre-registration.

**Amendment 2026-10-18 (user wording corrections; append-only; supersedes the matching phrases above)**
1. **Retracted:** "no held-out unit exists and any number is resubstitution only". Clip-level LOO can hold out a
   positive clip. What is unavailable is reliable validation, because n is tiny and there is no independent control,
   and any test against negative intervals.
2. **Scope of the conclusion.** An informative binary interval-level fit is impossible under the approved mask, which
   leaves 0 negative rows. The verdict does not claim that every kind of training is mathematically impossible.
3. **The 4/3 thresholds** are the operational screen of the locked stage-8 protocol. They are not a universal minimum
   for any research fit, and not a GO guarantee. Stage 5/7 GO bars are unchanged.
4. **Why the repeat clip-level experiment is rejected:** after the stage 4/5/7 NO-GOs on the same reused train-17, it
   would yield no new justified result. It could technically be fitted.

Verdict unchanged: NO_TRAINABLE_SUPERVISION for the proposed interval-level head.

## Stage 10 — supervision admissibility (critical-reviewer, claude-opus-5-5, 2026-10-18; read-only; no data, no caches, no synthetic run)
**Verdict: ADMISSIBLE as ONE pre-registered EXPLORATORY fit. It is weak, but no listed constraint is a logical or technical defect.** This narrows my stage-9 rejection. The new element is human-confirmed instance-level positives. Stages 4/5/7 used imported timestamps with ±windows, so this is not a pure repeat.
1. **Supervision types and allowed use.**
   - (a) The 2 confirmed intervals are instance-level positives under the strict mask (point-level frames, fixed inward margin on the uncertain sides).
   - (b) The 12 provider-explicit negatives are negative bags. Under the standard MIL assumption, every frame counts as a weak instance negative, with unknown provider label noise.
   - (c) The 3 provider-positive clips with unknown events may serve only as positive bags. They are excluded here.
   - NEVER: treat (c), or out-of-interval frames of any positive clip, as negative (in training or as FP); call (b) "confirmed normal windows/controls"; count frames as units (n = 14 clips).
2. **Constraints. All are weakening, none is inadmissible.**
   - **Scene identity (C5-1).** In each positive-held-out fold, the training positives come from ONE clip, so the label equals clip identity. A clip-level AUC cannot separate action from scene, camera or cashier. Only within-clip localisation on the held-out positive (scene held constant) speaks to the action.
   - **Duration.** Negatives are longer. A max-score is length-biased, so aggregate over a fixed K frames evenly spaced per clip. Duration-only AUC on the same 14 clips must be computed before lock. If it equals 1.0, no clip-level claim is attainable and the result is descriptive only.
   - **Pooled-LOCO AUC.** It mixes scores from 14 different models (known bias). Use leave-pair-out instead (2×12 = 24 pairs, each scored by one model).
   - **Tiny n.** Recall can only be 0/2, 1/2 or 2/2. FP 0/12 has an upper 95 % bound of 26 % (Clopper-Pearson). AUC = 1 has nominal chance 1/91 only under exchangeability, which scene, duration and reuse break, so no p-value is reported.
   - **Reuse and timing.** Train has been reused 5+ times, and the design came after stage-7 N1, so the run is exploratory only.
   - **Weighting.** Weight equally per clip and balance the classes, or the 12 long clips swamp the fit.
3. **Minimal pre-registration.**
   - Snapshot and cache shas; units deduplicated by CONTENT hash (one byte-identical negative pair → 12 units, never across folds); features (exact SigLIP2 zone cache, point-level).
   - Mask margin; whether the partial-visibility interval is included or weighted; clip and class weights; fixed λ; scaler fit on training clips only; K-frame aggregation.
   - A threshold rule fixed before scores (e.g. max training-negative-clip score).
   - Outcomes:
     - LPO-AUC vs duration-only AUC, plus clip F1 vs always-positive (0.25);
     - localisation per held-out positive (argmax inside the interval) vs a uniform reference (interval fraction, both: f1·f2) and the position prior;
     - in-interval vs out-of-interval score gap (descriptive; the outside frames are unknown, NOT negatives);
     - FP count /12 with its CI.
   - Run limits: one run, ≤ 5 min, locked hashes, aggregate summary 0400, weights not shipped, no tuning after scores.
   - Outcome labels: EXPLORATORY_TRANSFER_OBSERVED / NOT_OBSERVED. Never GO; production stays fail-closed; holdout CLOSED.
   - **A "hit" can mean only this:** on this reused train set, a direction fitted on one confirmed interval vs 12 provider-negative clips ranked another clip's confirmed interval above (i) that clip's own other frames and/or (ii) the provider-negative clips.
   - **It cannot mean:** a violation detector, generalisation, a validated FP rate or recall, significance, confirmed-normal negatives, or (without (i)) anything beyond scene similarity. A miss does not show absence of signal.
4. **Not applicable.** No irreparable defect was found. The only hard prohibitions are the NEVER items in (1).

### Stage 10 — pre-lock (critical-reviewer, claude-opus-5-5, 2026-10-18; read-only; synthetic only; no real run)
**Hashes recomputed and matching:** PROTOCOL_STAGE10 0334eaba0c796de8 (220 l); stage10_weak 3b6444cb06d3751b (498 l); stage10_selftest e95407ff1a56f538; `--print-lock` 4c1566d5ee6a63bf. Self-test **10/10 PASS** (39 s). pytest **172 passed**.

**Verified (code, file:line).**
- **Units and dedup.** Content-sha16 sets (S8.train_sets) deduplicate units. pos∩neg is refused (:330). The self-test turns 13 rows into 11 synthetic units, matching 13 → 12 on real data.
- **Leave-pair-out.** One model per pair, trained on the other units (:242). Leave-one-positive-out is used for localisation (:271), for 26 fits with no refits.
- **Fixed settings.** K = 20 grid max (:127, :167). Scaler on training clips' grid frames only (:155). Equal clip weights and balanced classes, Σw = n (:157). λ = S9.LAMBDA fixed. θ = max training-negative score (:246).
- **Clopper–Pearson** matches the reference values for n = 12: 0 → [0, .2646]; 1 → [.0021, .3848]; 6 → [.2109, .7891]; 12 → [.7354, 1].
- **Hits.** exact and tolerance hits each use a matched-support chance share (:202-205). The OBSERVED rule is identical in code (:208-213, RULE) and in protocol §5.
- **NEVER-items enforced.**
  - Unknown positives are never loaded or put in the manifest (:334, :359).
  - Outside-interval frames are never rows (S9.rows) and never FP; FP comes only from negative clips (:258).
  - No p-values are computed.
- **Integrity.**
  - Pins are checked before (:461) and after (:470).
  - Refusals return rc 2 with no summary (self-tested, including a tampered cache).
  - The child gets its own session, stderr DEVNULL, killpg at budget (:391-402). FAILED_* still writes a summary (O_EXCL 0400).
  - No weights are written, no leaks (leak_free), holdout chmod-000 spy passes.

**Findings.**
- **P10-1 MED (honesty of the template; corrects my own Stage-10 wording).** Clip-median centring removes only a static per-clip scene offset. Criterion (i) therefore does NOT exclude a time-varying activity or novelty pattern shared by the positive clips (cf. stage-7 N1). The self-test `scene` fixture, with a shared burst direction and no action cue, gives hits 0.35 vs chance 0.17 and false OBSERVED 0.10 vs reference 0.03. Exact edits:
  - Protocol §5 "Open risk": replace "…and only the within-clip criteria (i) speak to the action." with "The within-clip criteria (i) exclude only a static per-clip scene offset; they do not exclude time-varying activity or novelty shared by the positive clips (self-test `scene`: false OBSERVED 0.10; cf. stage-7 N1), so even OBSERVED does not establish violation specificity."
  - stage10_weak.py:55 TEMPLATE: after "without (i) it means scene similarity;" insert " with (i) it may still reflect generic within-clip activity or novelty shared by the positive clips, not violation specificity;".
- **P10-2 LOW (descriptive asymmetry).** The (ii) share at :252 compares the max over ALL in-interval frames with the negative's K-grid max, which favours positives. Add to protocol §5 (ii): "(asymmetric: all in-interval frames vs the negative's K = 20 grid; favours positives; descriptive)".
- **P10-3 LOW (interpretation, no change).** θ is the max of training-negative scores that the fit has suppressed, so held-out negatives exceed θ more easily. The FP count is pessimistic and is not a calibrated rate.
- **P10-4 LOW (optional).** The post-run `input_pins` at :470 is not guarded. If it raised, no summary would be written. This is unlikely, because the files are pinned.
- **Untested path.** The timeout kill is not exercised by the self-test. Line reading only.

**Verdict: REVISE (text only: P10-1, plus optional P10-2), then APPROVE FOR LOCK.** No change to the method, criteria, mask or inputs is needed. The lock digest changes with these edits, so lock the re-printed digest after a diff check. Holdout stays closed.

### Stage 10 — pre-lock diff check (critical-reviewer, claude-opus-5-5)
- **Hashes recomputed and matching:** PROTOCOL_STAGE10 40abb8cd9742df5e (222 l); stage10_weak f120d09fd218a8d9 (499 l); selftest e95407ff1a56f538 (unchanged); `--print-lock` **05cb78620b28a8a0**; inputs pin 735de57dbc2ee643 (unchanged).
- **P10-1 applied:** in protocol §5 l.158-162 and in TEMPLATE (stage10_weak.py:49-54), with exactly the requested wording.
- **P10-2 applied:** protocol §5 l.134-135.
- **P10-4 applied:** stage10_weak.py:467-471. A post-run pin failure now sets `comb1 = "error"`, so the summary is still written with `unchanged_after_run` false and `pins_after` "error". This is fail-visible.
- **Docstring:** text only.
- **Method unchanged:** λ, margin, K, tolerances, budget, θ rule and OBSERVED rule are all as before.
- **Self-test re-run:** 10/10, with aggregates identical to the pre-lock run.
- I did not diff the rest of the file byte for byte. There is no prior copy in git; the changed regions were inspected.

**Verdict: APPROVE FOR LOCK at digest 05cb78620b28a8a0.**

### Stage 10 — post-run conformity and mask audit (critical-reviewer, claude-opus-5-5; read-only; NO re-fit)
**Conformity: YES.**
- **Locked files:** chmod 444; sha16 40abb8cd9742df5e / f120d09fd218a8d9 / e95407ff1a56f538; `--print-lock` gives 05cb78620b28a8a0, the same digest as in the summary.
- **Summary:** mode 0400, 4955 B, sha16 64cc974e781ac337.
- **Integrity:** inputs 735de57dbc2ee643 verified and `unchanged_after_run` true (`pins_after` ok). Snapshot verified and unchanged. Stage-8 summary unchanged. Immutables 8/8 unchanged. `model_artefact_saved` false.
- **Content:** aggregates only. The only hex tokens are the six pins/lock. No slash, no Cyrillic, no times.
- **Outcome:** STOP_NOT_EVALUABLE follows the rule (protocol l.15; code :313). Pairs: 24/24 `no_positive_rows_under_strict_mask` (:150). Leave-one-positive-out: 2/2.

**Mask audit.** A local scratch script printed only buckets and booleans. It used the locked `units` / `clip_rows` / `S9.rows`.

| unit | rows, no margin | rows, after margin | width | uncertain sides | cache: first sample / step / n vs dur | interval vs samples |
|---|---|---|---|---|---|---|
| P1 | 0 | 0 | 0–0.5 s | 1 (start) | 0.25–0.75 s / 1.00 s / n≈dur | ends before the first sample; no intersection |
| P2 | 0 | 0 | 0–0.5 s | 1 (start) | 0.25–0.75 s / 1.00 s / n≈dur | ends before the first sample; no intersection |

- All 12 negatives also have first-sample offset 0.25–0.75 s.
- Both intervals have start > 0 and lie inside the decoded duration.
- Boundary values have ≤ 2 decimals. Typed input looks like that; button capture (`cts()`, annotate_intervals_ui.py:122) gives 3.
- This reproduces the user's diagnostic exactly.

**Causes (stated separately; this corrects the "margin" wording).**
1. **PRIMARY: interval and sampling mismatch.** No cached sample falls in either recorded interval even without a margin.
   - The cache is complete, so this is NOT a coverage gap. Decoding is 1 fps with t_k = k + 0.5 s (stage5_encoder.py:236, :263, :268), it starts at about 0.5 s, and n ≈ dur for all 14 units.
   - The cause is the label values themselves. Both are shorter than 0.5 s and end before the 0.5 s first sample.
2. **SECONDARY: margin collapse.** S9.rows shrinks inward (stage9_head.py:143: start + 5 if start-uncertain), which would also empty both intervals (width < 5 s). It is not the binding cause.

**Mask bug: NONE FOUND.**
- Sides and signs are correct (:143).
- Units are seconds on both sides: `_t1` checks 0 ≤ v ≤ dur in seconds (annotate_intervals.py:50-55), and the cache stores decode seconds idx/fps. A ms/s error would put the values above dur, not below 0.5 s.
- The mapping is by content sha16. Zero rows does not depend on which clip is used, because the values themselves are below 0.5 s.

**Finding P10-5 MED (label data quality, not code).**
- Both "confirmed intervals" are typed values, both under 0.5 s and at the very start of the clip.
- This is most consistent with an entry-unit slip: minutes or mm.ss typed into an unlabelled seconds field (annotate_intervals_ui.py:97-99, `type=number step=0.1`). Both reinterpretations fall within the clip duration (boolean only).
- It is NOT verified, and it must not be auto-reinterpreted, because that would be a new label. The annotator should re-check locally.
- UI follow-up for ui-dev: a unit label ("с"), a minimum width ≥ 1 sample step, and a start-near-zero warning.
- Any future gate should count only intervals that cover at least one cached sample.
- Consequence: my Stage-10 admissibility premise ("2 human-confirmed numeric intervals as instance positives") did not hold in practice.

**Vacuous defaults (no fits ran).** These must be shown as N/A, not as measurements. The locked summary is not rewritten.
- `clip_level.f1` 0.0 comes from the 0-denominator branch (:259), with tp=fp=fn=tn=0.
- `irls_converged_all` true is `all([])` (:312).
- `exact_interval_hits` [0, 0], `observed_rule_positives_passing` [0, 0] and the prior hits [0, 0] all have n = 0.

**Wording to confirm.**
- 12 provider-explicit weak negatives exist, and all 12 caches are valid.
- The stop is NOT a universal impossibility of training, and NOT NO_TRANSFER. It was caused by the 2 recorded intervals covering no 1-fps sample (primary), with margin collapse secondary. It was not caused by the margin alone.
- Duration-only separability 0.875 is a real measurement. A model comparison is unavailable.

**Coordinator note on P10-5 (user directive, 2026-10-18 ~04:48 UTC; appended, reviewer text above unchanged).** The minutes/mm:ss entry explanation is an UNCONFIRMED hypothesis only. There is no basis to claim the annotator entered wrong units, or that a short interval is invalid by itself. Verified fact: the recorded intervals contain no 1-fps cache sample (primary); the 5 s margin additionally collapses them (secondary). The UI-fix suggestions and the "minimum width ≥ 1 sample step" gate are NOT adopted. No re-annotation is requested. Stage 10 closed without a new stage, fit or UI change.
