# Cross review: claims vs code (cross-reviewer, sonnet)

Date 2026-10-17. Scope: claim-vs-code fidelity and integration seams (not the adversarial review).
Data boundary respected: synthetic data only, scratch dir `.../scratchpad/cross`; no real directories touched.
Own runs: `pytest -q tests -p no:cacheprovider` -> **87 passed, 1 skipped, 0 failed** (21.8 s); `python -m tools._selftest_ui` -> ok.

## 1. queue-architect claims

| Claim | Verdict | Evidence |
|---|---|---|
| Deadline from submission incl. queue wait | VERIFIED | `app/jobs.py:70,77` `t0 = submitted_at or time.time()`, `deadline = t0 + job_budget_sec`; worker rechecks at `jobs.py:160` |
| SIGKILL of process group | VERIFIED | `jobs.py:199` `start_new_session=True`, `jobs.py:381` `os.killpg(..., SIGKILL)` with `proc.kill()` fallback |
| Watchdog | VERIFIED | `jobs.py:239-261` (fails job at deadline + `kill_grace_sec`, kills proc, TTL expiry). Note: in-process `analyze_fn` mode cannot be killed (documented, tests only) |
| `QueueFull` subclass of `queue.Full`, 429 | VERIFIED | `jobs.py:40`; `main.py:173-175` |
| Cleanup touches only own patterns | VERIFIED | `_TEMP_RE`/`_RESULT_RE` fullmatch 32-hex (`jobs.py:36-37,322-328`); other names `continue` |
| `get()` disk fallback re-validated | VERIFIED | `jobs.py:290-302`: done -> `_done()` (validate_result), failed -> `_failed()` (safe_code), job id regex at `:277` |
| Error codes exactly the listed set | VERIFIED (with note) | Every `AnalysisError(` call in `app/pipeline.py` (lines 55,61,64,88,128,143,158,256,381,395-397,403,408-410,445) and `DetectorError` in `app/detector.py` (136,139,151,155,158,191,255) maps into `ALLOWED_ERROR_CODES` (`app/worker.py:50`). `time_budget` (pipeline.py:88) is not in the allow-list but is aliased via `_CODE_ALIASES` (`worker.py:39,82`) -> `time_budget_exceeded`, so no silent degradation. `model_load_failed` is the `DetectorError` default (`detector.py:28`, used at :151). Lines 395 and 408 are multi-line calls (code on next line) - fine. |
| Budget default 230, <300 check | VERIFIED | `config.py:24`; `config.py:59-60` rejects `upload_timeout + budget >= 300`; also `0 < budget < 300` (`:51`) |

Minor notes (not discrepancies): `config.Settings` field defaults are evaluated at import time (`config.py:19-44`), so env changes after import are ignored; `ruleset` default `"app/rules_default.json"` and `data_dir` `"data/jobs"` are CWD-relative (`config.py:20,29`, read at `pipeline.py:435`). Starting uvicorn from another directory makes every job fail (OSError -> not an `AnalysisError` -> `analysis_failed`). `pipeline.DEFAULT_RULESET` is absolute, but `Settings.ruleset` overrides it.

## 2. api-dev claims

| Claim | Verdict | Evidence |
|---|---|---|
| Key checked before body | VERIFIED | `main.py:79-81` from scope headers; 401 with 0 bytes read. (Route dependency `auth` at `:147` is a second check.) Test run live: no-key POST -> 401 |
| Content-Length early 413 | VERIFIED | `main.py:86-87` (cap = max_upload_mb + 1 MiB framing slack) |
| Byte cap during receive | VERIFIED | `main.py:101-105` |
| 408 timeouts (per chunk / total) | VERIFIED | `main.py:92-100` |
| No orphan upload/job | VERIFIED (code + tests) | `main.py:176-181` removes unsubmitted file; `submit()` removes file on rejection (`jobs.py:81-83`); tests in test_api.py pass |
| Static 422/500 bodies | VERIFIED | `main.py:139-145`; live: invalid POST returned `{"detail":"invalid request"}` |
| `/api/v1/model` content | VERIFIED | live (no SYNTHETIC_DETECTOR, `app.asgi`): `model{name,license,license_url,weights_sha256_prefix,validated,providers,input,rules[],allow_unvalidated_rules}`, `queue`, `limits`. Auth enforced (401 without key) |
| `store.close()` in lifespan | VERIFIED | `main.py:126-131` |
| `submitted_at` passed | VERIFIED | `main.py:153,171`; t0 recorded in `_Guard` at `:62-63` before key check/body |
| **Env vs config duplication** | **DISCREPANCY** | `main.py:136-137` reads `RECEIVE_TIMEOUT_SEC`/`UPLOAD_TIMEOUT_SEC` via `_env_f` (default 15/60) while `config.py:43-44` has `receive_timeout_sec`/`upload_timeout_sec` (default 15/60) and validates `upload+budget<300` (`:59`). Defaults agree. Divergence: (a) a `Settings(upload_timeout_sec=...)` passed to `create_app` (tests, embedding) is ignored by the guard; (b) a bad env value raises in config but silently falls back to default in main (`main.py:29-34`); (c) the `<300` invariant is enforced on config's value, not on the value the guard uses if they differ. Fix (owner api-dev): use `s.receive_timeout_sec` / `s.upload_timeout_sec`. |

## 3. ui-dev claims

| Claim | Verdict | Evidence |
|---|---|---|
| Inventory prints no names | VERIFIED with one leak | Synthetic tree with `ZZSENTINEL_dir`, nested `sub_ZZSENTINEL`, `ZZSENTINEL_clip1.mp4`, `ZZSENTINEL_labels.csv`, csv column `ZZSENTINEL_col`, json key `ZZSENTINEL_key`, values `ZZSENTINEL`: stdout of default, `--probe` and `--labels C01` contained **no** sentinel in names, dirs, columns or values (column names filtered by `NAME_ALLOW`). Map went to the scratch private dir (I patched `write_map`); the repo's `local_private/` was not created. **Leak:** `inventory.py:205` prints the lowercased file extension verbatim; a file `x.ZZSENTINEL` printed `.zzsentinel=1`. Extensions of real files are arbitrary strings (low risk, but a free-text channel). Suggest allow-listing known extensions and bucketing the rest as `other`. |
| Annotate loopback-only | VERIFIED | binds 127.0.0.1 (`annotate.py:378`); middleware checks client IP and Host (`:151-157`) -> 403 (blocks DNS rebinding). Edge: Host `::1` without brackets/port parses oddly (fails closed). |
| Range endpoint | VERIFIED | `annotate.py:205-242`: 200/206/416, `Accept-Ranges`, `Content-Range` |
| `clip_status` only by human | VERIFIED | written only in `PUT .../status` (`annotate.py:279-290`); `clean_status` rejects other values; "unreviewed" = no record. Selftest "9/9 groups" (claim said 8/8; newer, not a defect) |
| index.html fields exist in `/api/v1/model` | VERIFIED with note | `static/index.html:35-46` uses `model.name`, `.license`, `.error`, `.rules[].type/.status/.validated/.active`, `.allow_unvalidated_rules` - all present in live response. `mi.available` (`:38`) is **absent** from the success response; it exists only in `main.py:199` fallback `{"available": False}`. Works because `mi.error` covers the weights-missing path (`detector.py:274`), but the contract is implicit. Rule `status` values `candidate`/`validated` match `TXT` map (`index.html:30`). |

## 4. test-dev claims

| Claim | Verdict | Evidence |
|---|---|---|
| 87 / 1 skipped / 0 failed | VERIFIED | own rerun, identical |
| `SYNTHETIC_DETECTOR` only in tests; production has no default | VERIFIED | set only in `tests/conftest.py:13` (monkeypatch, autouse); `app/asgi.py` is `create_app()` only; `detector.py:241` honours it only if env var is `1`. Live run with the var unset gave real `yolox_s` |
| **Second-person negative test would fail if the rule were enabled+opt-in** | **DISCREPANCY (test is vacuous)** | `tests/test_pipeline_rules.py:70-78` already enables SECOND + opt-in, but I mutated a scratch copy of the repo (`.../cross/mut`: `NEVER_VERDICT = set()` and `observation_only` removed from the ruleset) and **all 21 tests in test_pipeline_rules.py still pass**. Cause: on the synthetic `two=(5,35)` clip the pipeline emits **no** second-person observation at all (`analyze()` returned `observations: []`, verdict `no_violation`), so `all(e["type"] != SECOND for e in violations)` is checked over an empty list and the following `for e in j["violations"]` loop never runs. The hard-coded protection (`pipeline.py:324,334`) is therefore not regression-protected. Owner test-dev: make the synthetic two-person fixture actually trigger the observation (assert SECOND appears in observations/`ob["scored"]`), then assert it is excluded from violations. Likely also `test_default_rules_two_persons_is_failed_not_violation` passes for the wrong reason (no active rule), not because of 2 persons. |

## 5. Integration seams

| Seam | Verdict | Evidence |
|---|---|---|
| norms.py vs norms_source_excerpt.md numbering | VERIFIED | п.20, 22, 24(3-4), 25, 31(4), 32, 42, 75 all exist in the excerpt with the same numbers; п.80 is in the excerpt but unused by norms.py |
| No invented wording | PARTIAL / DISCREPANCY (wording drift) | `norms.py:15` p42 summary "контролёр при пересчёте/сортировке/упаковке" omits the official exception "за исключением ... приему и выдаче наличных денег клиенту" (excerpt, п.42) - this exception is exactly why second person != violation; `norms.py:18` p75 "доступ в хранилище при наличии полномочий/сопровождения" paraphrases "кладовая и (или) сейфовая комната", "лица, ответственные за сохранность", "на основании распорядительного документа ... при сопровождении" loosely. `norms.py:1-6` docstring still says "NOT verified against the current official text (F4)" - stale, F4 was closed by norms-fetcher. Owner detection-architect. |
| `describe_model()` fields vs UI | VERIFIED | see item 3 (`pipeline.py:73-83`, `rule_states` `:327-340`) |
| Observations export path vs annotate | VERIFIED (path/format) / GAP (no entry point) | `pipeline.export_observations` (`pipeline.py:481-499`) writes `local_private/observations/<sha256(file)[:16]>.json` as `[{type,start_sec,end_sec,confidence}]`; `annotate.read_observations` (`annotate.py:132`) reads the same dir/format and `sha_prefix` = sha256 hex[:16] (`annotate.py:33-38`) - identical definition. But `export_observations` has **no caller anywhere** (no CLI, script or doc), and its default `out_dir` is CWD-relative. The "accept/reject observation" UI therefore shows nothing unless someone runs it manually. |
| evaluate.py labels vs annotate records | VERIFIED | annotate writes `{"kind":"event","sha","t"}`, `{"kind":"clip_status","sha","status"}`, `obs_review` (`annotate.py:87-116`); evaluate reads exactly these (`evaluate.py:31-62`), ignores `obs_review`, falls back to `sha16`/`timestamp_sec`; statuses identical (`evaluate.py:23`). Predictions must be `DIR/<sha16>.json` with the API body incl. `violations[].timecode_sec` (`evaluate.py:93-109`) - matches `worker.py:213`. **Gap:** no tool produces `<sha16>.json` prediction files from local clips (runner that submits clips to the API/pipeline and names outputs by sha16). |

## 6. Docs / licences

| Item | Verdict | Evidence |
|---|---|---|
| LICENSES.md versions vs installed | VERIFIED | fastapi 0.143.3, uvicorn 0.54.0, opencv-headless 5.0.0.93, numpy 2.5.4, onnxruntime 1.31.0, python-multipart 0.0.32 match `.venv` dist-info |
| Missing rows | DISCREPANCY | installed/required but absent: openpyxl 3.1.5 (requirements.txt), certifi (requirements.txt), starlette 1.8.0 (transitive), uvicorn[standard] extras |
| Weights row | DISCREPANCY | still placeholder (`LICENSES.md` last row "to be filled by detection-architect"). Facts available: YOLOX Apache-2.0, `/api/v1/model` reports sha256 prefix `c5c2d13e59ae` (yolox_s), `weights/` has yolox_s/tiny/nano .onnx; `detector.LICENSE_URL` exists. Owner docs-writer/detection-architect |
| requirements.txt vs COORDINATION claim "onnxruntime==1.31.0 + certifi pinned" | PARTIAL | onnxruntime pinned; `certifi`, `numpy`, `uvicorn`, `python-multipart`, `openpyxl` unpinned (coordinator) |

## Integration gaps (owner)

1. test-dev: second-person negative test is vacuous (no SECOND observation produced); mutation survives. Highest priority.
2. api-dev: guard reads timeouts from env, not from `Settings` (`main.py:136-137`).
3. detection-architect: no CLI/entry point for `export_observations`; `out_dir` CWD-relative; norms.py p42 summary omits the client-operation exception; stale F4 docstring.
4. test-dev / ui-dev: no producer of `<sha16>.json` predictions for `tools/evaluate.py`.
5. ui-dev: `inventory.py:205` prints raw extensions (free-text channel); use allow-list.
6. queue-architect (config): CWD-relative `ruleset`/`data_dir` defaults; env read at import time.
7. docs-writer: fill weights row, add openpyxl/certifi/starlette.
8. static/index.html relies on `error` instead of an explicit `available` field; API should return `available` consistently (api-dev + detection-architect).

Model: I believe I run as claude-sonnet-5-5 (Sonnet 5.5), per my system prompt; unverified by API.

---

# Pass 2 (waves 5/6 fix claims)

Own runs: `pytest -q tests -p no:cacheprovider` -> **131 passed, 0 failed, 0 skipped** (29.9 s). `python -m tools._selftest_ui` -> **12/12 groups passed**. All synthetic, scratch dir `.../scratchpad/cross`; nothing run against `local_private/` or `~/Downloads`.

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | H2 `holdout_ok` | VERIFIED as written, WEAK in two ways | `app/detector_rules.py:40-54`: n_clips int >=5, p/r finite in [0,1], `tool == "tools/evaluate.py"`, non-empty `split_id`, ISO date. Bypass A: `evaluated_at: "2026-10-17junk"` is accepted (`str(...)[:10]` at :52 ignores the tail; `"2099-01-01"` also accepted). Bypass B: `precision=0, recall=0` is accepted, so a rule with zero measured quality can be "validated". Any hand-written JSON satisfies the record; nothing ties it to a real evaluate run (no hash of predictions/labels). |
| 1 | H3 floors | VERIFIED | validate() rejects person ratio <0.10 (also NaN), occupancy <0.15, min_conf outside [0,1] (bool rejected), `max_events` not int 1..50. Re-applied at use sites: `pipeline.py:355` `max(..., FLOOR_PERSON_RATIO)`, `:369` occupancy, `:436-437` emit threshold + `max_events` slice. Not covered by the claim and not validated: `rules.*.min_sec` (0 or missing accepted by validate; 0 would fail later, not at validation), `candidate_fraction_of_min_sec` (=0 accepted), `min_gap_between_events_sec` (-5 accepted). Fail-closed in practice, but not "invalid_config". |
| 1 | M1 unknown/missing rule keys | VERIFIED for rule types; extra top-level keys and unknown keys *inside* a rule entry are accepted (`validate`, :64-66 checks only `rules` names). |
| 2 | M5 `quiet_native_stderr` | VERIFIED | `detector.py:273-297`; used at `pipeline.py:409` (analyze) and `:483` (export). Canary: random-bytes `ZZSENTINEL_bad.mp4`, `analyze` and `export_observations` both raise `corrupt_video` and fd 2 captured **0 bytes**; control (plain `cv2.VideoCapture` on the same file) printed the full path to stderr, so the canary is meaningful. |
| 2 | M6 describe codes only | VERIFIED | `pipeline.py:82` `rules_error = e.code`; `RulesError` messages are static; ruleset_unreadable/invalid_config only. |
| 3 | norms.py wording | VERIFIED | п.42 now includes the exception "за исключением операций по приёму и выдаче наличных денег клиенту" (matches excerpt); п.75 matches excerpt wording (ответственные лица / распорядительный документ / сопровождение); п.20, 22, 24-25, 31-32 faithful condensations. SOURCE "ред. 12.07.2026, получено 2026-10-17" agrees with excerpt (/history last row 12.07.2026; the page itself carries no "редакция от" sentence - the date is derived, as the excerpt says). Stale F4 docstring is gone. |
| 4 | run_local.py | VERIFIED | Child per clip in own session (`:93-96`), stdio DEVNULL, SIGKILL of the group on timeout (`_kill_group`, `:84-89`, `:99-103`), child output re-validated with `worker.validate_result` / `safe_code` (`:105-114`). Synthetic run (candidate dir and files named ZZSENTINEL*, one 2000-byte garbage .mp4): stdout was counts only (0 sentinel hits); `--dry-run` printed counts and wrote nothing; both split formats (`{"clips":{sha:side}}` and `{"holdout":[sha]}`) selected the expected clip; `--budget-sec 1` refused (rc 2); prediction body `{"status":"failed","error":"no_validated_rule"}` (default rules); no `.child` temp left. The kill-at-budget path was verified by reading only (clips too fast to time out). Note: `LOCAL_PRIVATE_DIR` is honoured by run_local but not by `import_labels` / `evaluate --private` default (inconsistent, minor). |
| 5 | import_labels `--profile` | VERIFIED | Synthetic xlsx with Cyrillic sentinel headers/cells (`ЗЗСЕНТИНЕЛЬ_*`, file name `ZZ*`): profile and import stdout contained none; only per-column class counts. |
| 5 | ru token lists / compound / empty | VERIFIED | `import_labels.py:28-29` RU_POS / RU_NEG sets are exactly the lists in COORDINATION "SCHEMA RESULT"; seconds regex and `N мин N сек` (seconds<60) match. "5 сек" -> 5.0 s event; "нарушение" with "1 мин 5 сек" on a 13 s clip -> `imported_needs_user time_exceeds_duration` (not silently imported); empty label cell -> `unknown` (never `reviewed_no_violation`). `--import --auto-ru` counts line: rows=4 matched=4 events=1 statuses_set=3 needs_user=1 positive_with_time=2 n_time_within_duration=1 n_time_exceeds_duration=1, consistent with the records written. |
| 5 | evaluate `--source imported_dataset` | VERIFIED | Header line exactly "source: imported_dataset — по поставленной XLSX, без ручной проверки". With human `clip_status`/`event` records for a holdout sha in the same file: ignored (clip stays unreviewed); `unknown` excluded and counted; duplicate identical imported lines collapse; code path for conflicting duplicates -> `conflict` -> excluded as inconsistent. Directory listing identical before/after (writes nothing). Minor: a `conflict` clip counts as "reviewed" toward `--min-holdout` (`evaluate.py:187-190`), slightly inflating the minimum check. |
| 6 | Docs: flags / env vars | VERIFIED | Every `--flag` and UPPER_CASE name in README/ARCHITECTURE/QUALITY/DEPLOY/LICENSES exists in code, except external ones (`--host`, `--workers` uvicorn; `--find-links`, `--no-index` pip) and doc names (`REVIEW_CRITICAL`). `inventory ... --probe C02`, `annotate --port 8790`, `inventory.py` extension allow-list (`other_count`, `inventory.py:241-242`; fixes my pass-1 leak) all confirmed. |
| 6 | Docs: no clip names / label values | VERIFIED | Only aggregates and candidate IDs (C02, 42 clips, 22/20 split of labels, resolution, duration range). No file names, paths other than `~/Downloads` in command examples, no label text. |
| 6 | Docs: stale numbers | **DISCREPANCY** | `README.md:147-148`, `docs/QUALITY.md:35-37,109` still state "95 passed" and "90 passed, 5 failed" (and a "discrepancy 95 vs 90+5" item); actual: 131/0/0. Selftest count is not quoted in these docs. |
| 6 | Docs: workflow gaps | DISCREPANCY (minor) | README step 6 / QUALITY.md:78-79 show only the human-label `evaluate` call; the `--source imported_dataset --cid C02` mode that the coordinator actually uses is not documented, nor that `evaluate.py` flags `--source/--cid/--private`. LICENSES.md matches installed versions (certifi 2026.7.22, playwright 1.64.0, weights sha prefix c5c2d13e59ae confirmed on disk); weights row is now filled; uvicorn[standard] extras explicitly unreviewed. |

## Pass 2 gaps (owner)

1. detection-architect: tighten `holdout_ok` (strict ISO date, no future date, `precision`/`recall` not both trivially 0, ideally a hash tying the record to the predictions/labels it came from); validate `min_sec`, `candidate_fraction_of_min_sec`, `min_gap_between_events_sec` and reject unknown keys inside rule entries.
2. docs-writer: patch test counts to 131/0/0 and selftest 12/12 (README:147-148, QUALITY:35-37,109); document `evaluate --source imported_dataset --cid`.
3. test-dev: `conflict` clips should not count toward the `--min-holdout` check; add tests for `evaluated_at` junk-suffix and p=r=0.
4. ui-dev: `import_labels` should honour `LOCAL_PRIVATE_DIR` like `run_local` (it uses the fixed `inventory.PRIVATE`).

Model: I believe I run as claude-sonnet-5-5 (Sonnet 5.5), per my system prompt; unverified by API.

---

# Stage 3 licences (independent check, 2026-10-17)

Primary sources fetched with curl (HTTP 200) from the Hugging Face API/raw files at the pinned commits and from GitHub raw `LICENSE` at the pinned tags; installed METADATA read from `.venv-research`. Nothing was downloaded beyond licence/card text and API metadata; no model was run. Not legal advice.

| Component | Pinned | Licence (quoted) | Source URL | Commercial use OK? | Notes |
|---|---|---|---|---|---|
| Qwen/Qwen2-VL-2B-Instruct | rev 895c3a49 (full `895c3a49bc3fa70a340399125c650a463535e71c`) | model card front matter `license: apache-2.0`; API `cardData.license: apache-2.0`, tag `license:apache-2.0`, `gated: False`; `LICENSE` file (201 lines) = "Apache License Version 2.0, January 2004" | https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/895c3a49/LICENSE and `.../README.md` | yes | LICENSE contains no "Tongyi Qianwen" / "monthly active users" text (grep count 0), so the Qwen licence terms that apply to the larger Qwen2-VL-72B do not apply to this 2B repo at this commit. |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | rev 01af461c (full `01af461cdb9574acc09084a0ef94e216e142b085`) | card front matter `license: apache-2.0`; API `cardData.license: apache-2.0`, `gated: False` | https://huggingface.co/mlx-community/Qwen2-VL-2B-Instruct-4bit/resolve/01af461c/README.md | yes (inherited) | No LICENSE file in the repo (files: README.md, tokenizer/config files, `model.safetensors`, index); no added terms; card says converted from the Qwen repo with mlx-vlm 0.0.13 (card link target is empty). Licence is the declared one, derived from the base model's Apache-2.0. `model.safetensors` LFS oid/size from HF API = 3534d3dc...cf992 / 1,245,868,907 B, identical to `tools/fetch_weights.py`. |
| mlx | 0.32.4 | "MIT License ... Copyright © 2023 Apple Inc."; METADATA `License: MIT` | https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/LICENSE | yes | |
| mlx-vlm | 0.7.8 | "MIT License ... Copyright © 2025 Prince Canuma"; METADATA `License: MIT` + MIT classifier | https://raw.githubusercontent.com/Blaizzy/mlx-vlm/v0.7.8/LICENSE | yes | |
| transformers | 5.20.1 | "Apache License Version 2.0" (LICENSE at tag); METADATA `License: Apache 2.0 License` | https://raw.githubusercontent.com/huggingface/transformers/v5.20.1/LICENSE | yes | |
| huggingface-hub | 2.2.0 | "Apache License Version 2.0"; METADATA `License: Apache-2.0` | https://raw.githubusercontent.com/huggingface/huggingface_hub/v2.2.0/LICENSE | yes | |
| safetensors | 0.8.0 | "Apache License Version 2.0"; METADATA has only the classifier `License :: OSI Approved :: Apache Software License` (no License/License-Expression field) | https://raw.githubusercontent.com/huggingface/safetensors/v0.8.0/LICENSE | yes | |
| tokenizers | 0.23.3 | "Apache License Version 2.0" (repo root LICENSE at tag); METADATA has only the Apache classifier | https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.3/LICENSE | yes | `tokenizers/LICENSE` path is not at that tag; the root file was used. |
| pillow | 13.0.0 | METADATA `License-Expression: MIT-CMU`; LICENSE at tag is the HPND-style "The Python Imaging Library (PIL) is Copyright © 1997-2011 by Secret Labs AB ..." text | https://raw.githubusercontent.com/python-pillow/Pillow/13.0.0/LICENSE | yes | MIT-CMU / HPND is a permissive licence; I did not compare the text word-for-word against the SPDX MIT-CMU text. |

Training-data caveat (vendor): the Qwen2-VL card states "Data timeliness: Our image dataset is updated until June 2023" and lists limited recognition of specific individuals/IP, weak counting and weak 3D spatial reasoning; it does not publish a dataset licence or list of sources in the card (UNVERIFIED: the data terms of the underlying training sets; same caveat as COCO for YOLOX, so "Apache-2.0 weights" does not certify the training data).

Consistency of repo files:
- `tools/feature_models.py:4-7` states Qwen2-VL-2B-Instruct Apache-2.0, LICENSE file at `895c3a49`, 4-bit conversion at `01af461c`, fetched with SHA256 via `tools/fetch_weights.py qwen2_vl_2b_instruct_mlx4`, venv versions mlx 0.32.4, mlx-vlm 0.7.8, transformers 5.20.1: all agree with the primary sources and installed dist-info (also huggingface_hub 2.2.0, safetensors 0.8.0, tokenizers 0.23.3, pillow 13.0.0 installed).
- `tools/fetch_weights.py:10-12` and `HF_SNAPSHOTS["qwen2_vl_2b_instruct_mlx4"]` (:42-56): repo, revision `01af461cdb9574acc09084a0ef94e216e142b085` (matches API sha), 12 files with exact sizes and per-file SHA256; download verifies size and hash and rejects mismatches (`:61-74`). The docstring claim "the conversion's card declares apache-2.0 and ships no LICENSE file" is correct. Only `model.safetensors` hash was cross-checked with the HF LFS oid; other (small, non-LFS) file hashes not independently recomputed (UNVERIFIED).
- Gap: `docs/LICENSES.md` should get rows for the research venv components above (confirmed: `docs/LICENSES.md` currently has no Qwen/mlx rows).

Model: claude-sonnet-5-5 (Sonnet 5.5) per my system prompt; unverified by API.

## Stage 4 (cross-reviewer, claude-sonnet-5-5)
Claim-vs-code check of tools/stage4_probe.py, stage4_selftest.py, PROTOCOL_STAGE4.md. Synthetic only; real audit files hashed only.

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | 224 configs; features match cache | VERIFIED | stage4_probe.py:31-35 gives 14x2x2x2x2 = 224 (runtime len(CFGS)=224); all 14 names present in TP.features() output of a synthetic train_probe cache (22 keys) |
| 2 | Inner clip-LOO, F1 -> hit -> AUC -> lowest index, refit, one outer prediction | VERIFIED | :103-121. Inner threshold on 15 (:113-114), lexsort at :118 (f1 primary, hr, auc, -index), refit on 16 (:119). Outer clip k only enters at :120 as S[c,k]. Instrumented: 289 fit_thr calls per nested run = 17x(16+1) |
| 3 | Scanner label-free, earliest argmax, offset on scan point | VERIFIED | :50-72: end=t+off, argmax first index (:71), returns t[k] (scan point, not shifted). No label argument |
| 4 | Metrics via match_events/_prf, hit rate, baselines, permutations | VERIFIED with notes | train_probe.py:223-236 uses evaluate.match_events/_prf at TOL 5; hit_rate :130-133; random_time :170-185; duration baseline uses same nested() (:340-343); permutations call full nested() (:143-148); n_perm=5 -> 1734 fit_thr calls = 6 x 289 (primary + 5) |
| 5 | Self-test | VERIFIED | n_perm=50: planted GO (F1 1.0, hit 1.0), nosignal NO-GO, missing PASS (decoded 1); 3/3, 11 s total; holdout mode 000 + open() spy (stage4_selftest.py:98-116) |
| 6 | Privacy | VERIFIED | summary: no '/', no media ext, no Cyrillic; ">=16 hex" regex only matched decimal digits of floats (false positives), no sha. Zone kept in memory only (train_probe.py:176, not in stage4 res). Protocol hash is the only hash printed |
| 7 | Protocol hash | VERIFIED | sha256-16 of docs/PROTOCOL_STAGE4.md = 673a3c005c8ca67a (unchanged since run) |
| 8 | Line counts, pytest | VERIFIED | stage4_probe 423, selftest 151, protocol 163 lines; pytest 131 passed. attrs and R1 sha: unchanged |

### Implementation bugs
1. tools/evaluate.py:171 (owner test-dev): `f = ... if p is not None and r is not None and p + r else None` returns None when P=R=0. Correct rule: F1 = 0 when TP+FP+FN > 0; None only if all three are 0. Fix: `f = (2*p*r/(p+r) if p+r else 0.0) if (tp+fp+fn) else None` (p,r treated as 0 when their denominators are 0 but tp+fp+fn>0). Affects the recorded ablation event block (TP0/FP10/FN5 -> F1 null, should be 0.0). Primary all-17 metrics unaffected: clip TP3/FP7/FN2 F1 0.4 and event TP1/FP9/FN4 F1 0.1333 are computed with p+r>0. Decision logic uses clip F1 only (stage4_probe.py:352-356), so GO/NO-GO is unaffected. Do not re-run; keep stage4_summary.json as evidence, re-derive from recorded counts in a note.
2. stage4_probe.py:350 (owner det-stage4): ablation permutation seed is 1; protocol §5 says seed 0. Ablation is sensitivity only, so minor.
3. stage4_probe.py:166-167: RuntimeWarning "All-NaN slice" when a chosen score is constant (duration |rho| NaN); the NaN is silently carried. Protocol's |rho|>=0.5 duration-dependent flag is not computed, only median/max printed (protocol §5).
4. Doc gap: nosignal synthetic reaches F1 0.75 by chance, so the F1 criteria are weak at n=17 (already disclosed by det-stage4).

## Stage 5 — Part A (cross-reviewer, claude-sonnet-5-5, 2026-10-18)
Scope: licences from primary sources, fetch_weights siglip2 entry, encoder privacy, pins, protocol consistency. Synthetic only; real data not opened. Immutable files (hash only): attrs ab7a86b1…896f unchanged, R1 09247337…ac35 unchanged, map e413108f8b7f0080 unchanged, stage4_summary 284e70c08e70c02b unchanged.

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | siglip2 @75de2d55: licence apache-2.0, not gated, model_type siglip, no auto_map, no LICENSE file | VERIFIED | HF API revision JSON: sha 75de2d55…, gated false, cardData license apache-2.0, siblings = .gitattributes, README, config, safetensors, preprocessor, special_tokens_map, tokenizer.json/.model, tokenizer_config (no LICENSE); raw config.json: model_type siglip, no auto_map; card: "pre-trained on the WebLI dataset" (data terms unverified: caveat present, PROTOCOL_STAGE5.md:§10, fetch_weights.py:15) |
| 2 | big_vision@0127fb6b Apache-2.0; transformers v5.20.1 Apache-2.0; torch v2.14.1 BSD-style; onnx v1.23.1 Apache-2.0; onnxscript v0.7.2 MIT | VERIFIED | raw LICENSE headers fetched at the pinned tags (torch text is 3-clause BSD, not named in file) |
| 3 | fetch_weights.py:45-56 siglip2: 8 files with sizes+SHA256 | VERIFIED | recomputed all 8 sizes and SHA256 against the table: all match; model.safetensors 612923381c76ec5a, onnx/vision.onnx 4e95b080c23eff44; 0 .part/.incomplete files under weights/ |
| 4 | encoder: no trust_remote_code, local_files_only, stdin jobs, stderr DEVNULL, no printing of frames/paths | VERIFIED | stage5_encoder.py:101,102,127 local_files_only=True; no trust_remote_code anywhere; :222-223 stdin PIPE, stderr=DEVNULL; only 3 prints (:246,248,251), numeric aggregates and a model-dir size. Smoke rc=0 (16 frames, LOCAL_PRIVATE_DIR in scratch): ORT CPU 37 ms/frame, torch MPS 15-22 ms, cos vs torch 1.0, preproc cos 0.992, peak RSS 2.0 GB; output grep for paths/Cyrillic: 0; "hex>=16" hits are float digits (false positive) |
| 5 | requirements-research.txt pins == installed .venv-research | VERIFIED | torch 2.14.1, onnx 1.23.1, onnxscript 0.7.2, transformers 5.20.1 and all stage-3 pins match; ORT 1.31.0 in main venv == requirements.txt:6 |
| 6 | protocol: labels only in training targets; 24-grid; six GO criteria; disclosure | VERIFIED | §3 targets t in [label-10,label+2] on training positives only; §5.5 label isolation + perturbation test; localisation rules label-free; grid 2x2x3x2=24 with 12 distinct fits (§4); budget 1850 fits = 136 pairs x12 + 17x12 refits, consistent; §7 lists 6 criteria incl. duration-only and |rho|<0.5; disclosure at top and §0 |

### Discrepancies / bugs
1. tools/stage5_encoder.py:165-181 (owner det-stage5): `smoke` re-exports ONNX over the pinned weights/.../onnx/vision.onnx every run. This run the SHA stayed 4e95b080c23eff44 (deterministic here), but a later torch/exporter change could silently alter the pinned derived file. Suggest exporting only if missing, or to a temp path and comparing.
2. docs/LICENSES.md has no SigLIP2/WebLI/big_vision/torch/onnx entries yet (grep siglip: 0 hits); PROTOCOL_STAGE5.md §10 holds them. Owner docs-writer.
3. stage5_encoder.py:251-252 prints a directory size labelled "external data if any" (no external data exists); cosmetic. Key name `ort_CoreMLExecutionProvider_error` (:215) differs from the `ort_coreml_*` pattern; cosmetic.
4. Not in code yet: tools/stage5_probe.py / selftest were not present at review, so protocol §5/§6/§9 implementation (label isolation, permutation, holdout 000 spy) is UNREVIEWED here (Part B).

Open risks: WebLI data terms unverified (disclosed); zone view depends on stage-4 cache being label-free (not re-audited here).

## Stage 5 — Part B (cross-reviewer, claude-sonnet-5-5, 2026-10-18)
Scope: claim-vs-code of tools/stage5_probe.py (338 lines), stage5_model.py (237), stage5_encoder.py (445), stage5_selftest.py (288) against docs/PROTOCOL_STAGE5.md (sha256-16 b0d73d7ab46da775 = confirmed). Synthetic only (scratch cross5b); the real run was not repeated; videos, frames and human answers not opened. Immutables (hash): attrs ab7a86b1…896f, R1 09247337…ac35, map e413108f, stage4_summary 284e70c0 — all unchanged.

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | 24-grid, 18 features, PCA-8 in-fold, 8 prompts | VERIFIED | stage5_model.py:15-19 (2 views x 2 families x 3 lambda x 2 loc = 24; 12 fits); 18 = lev 8 + residual 8 + change + dist-to-median (:55-58, :40-52); PCA fitted on the training set only, FoldCache.X :67-76; W 5 s, MIN_VALID 3 (:28-32); 8 PROMPTS stage5_encoder.py:37-46 |
| 2 | Nesting: inner clip-LOO, J = F1 + hit, AUC, index, refit on 16, one outer prediction | VERIFIED | model.py:144-167, 174-204. Fit-count instrumentation on a 6-clip fixture: 252 fit calls = 12 x C(6,2) + 12 x 6; train sizes {4: 180, 5: 72}; held-out clip never in its own training set (0 leaks); 21 distinct training sets x 4 (view, family) = 84 cache keys. Outer-clip label flip changes nothing in that clip's prediction. Key (f1+hr, auc, -c) = tie-break by lowest index (:198). Inner threshold excludes j (:192-194); outer threshold over the 16 inner scores (:201) |
| 3 | Scanner label-free; peak / run_end | VERIFIED | localise (model.py:131-140) takes only t, logit, valid; earliest argmax; run extends forward while valid and logit >= max - 1.0 (the run end is the same as for the maximal run) |
| 4 | Metrics via evaluate.match_events/_prf (post-fix), hit +-5 s, baselines and permutations full nested | VERIFIED | train_probe.py:223-236 uses evaluate._prf (evaluate.py:171 post-fix F1 = 2TP/(2TP+FP+FN)); model.py:15 TOL 5; probe.py:151-161 each replicate calls full M.nested; permute carries relative label positions (model.py:222-226); duration-only via S4.nested (probe.py:181-188) |
| 5 | 90-min global guard, decode_incomplete fail-closed, memory watchdog, zone fallback | VERIFIED (code), PARTIAL (tests) | Guard: probe.py:91-99 (remaining < 30 s -> incomplete_budget, counted as failure), :152 (perm loop stops if now + 2 x max run > deadline), :257, :262. decode_incomplete: encoder.py:258-278, run_watched maps to status (:370-371); selftest "corrupt" scenario passes (decode_incomplete 1, units 16, no cache, no GO). Watchdog: `memtest --mem-mb 150` -> status=memory_cap, peak 154 MB, 0 leftover children; cap 8192 -> ok, peak 330 MB (ps -g works on this macOS). Zone fallback counted probe.py:122-123 and tested (zone_fallback_1 in every scenario). NO synthetic test exercises the global-deadline path itself (see bug 2) |
| 6 | stage5_summary.json aggregates only | VERIFIED | Key structure: protocol, protocol_sha256_16, coverage{counts, p50/p95/max lists, mem}, primary_all_units{outer clip/event, baselines, perm{n, n_completed, complete, seed, f1, hit_rate}, chosen_configs{distinct, top-6 config NAMES, per-dimension counts}, rho, run-time}, baseline_duration_only_nested, duration_confound, SENSITIVITY_view_full_only, zone_groups{agree, mismatch, no_r1}, ABLATION..., decision{6 criteria + primary_complete + GO}, incomplete_phases, immutable_files{n, status}, runtime_sec, parent_maxrss_mb, disclosure. Scan: 0 '/', no media ext, no Cyrillic, no prompt text; the only 16+ hex string is the public protocol sha256-16. No per-clip rows |
| 7 | selftest 5/5 reproducible at n_perm 50 | VERIFIED | planted PASS (F1 0.889, hit 1.0, GO), perturb PASS (34 outer perturbations, 0 changes; inner flip changed 15), null NO-GO, long GO via run_end (17/17), corrupt NO-GO; "5/5 passed". Runtime 2533 s CPU, 14 min 29 s wall (the claimed 718.8 s is the real run, not the selftest) |
| 8 | Part A finding: export ONNX only if missing + hash check | VERIFIED | encoder.py:169-170 export loop is skipped if the file exists (onnx_export = kept_existing); onnx_ok() sha256 compare (:223-232) is used by the embed child (:249) and by probe.py:83. The file sha still starts 4e95b080c23e. Residual: child_ort (the smoke) does not check the hash (:205) |
| 9 | Line counts < 500; pytest 134 | VERIFIED | probe 338, model 237, encoder 445, selftest 288; protocol 330 lines; pytest: 134 passed, 1 warning (44 s) |

### Bugs / discrepancies
1. tools/stage5_probe.py:264-265 (owner det-stage5): `elif agree.sum() >= 6` records "ablation" as an incomplete phase also when the agree subset has a single class (not evaluable, nothing timed out). Label is misleading; no effect on the decision.
2. No test of the global-deadline path (probe.py:91, :152, :257, :262) and no pytest for stage5 modules (tests/ has no stage5 file); the guard was reviewed by reading only. Owner test-dev / det-stage5. The run finished in 718.8 s of 5400 s, so the guard was not exercised for real.
3. model.py:229-233 `null_summary([])` (zero completed permutations) returns nan p50/p95 with a numpy RuntimeWarning on stderr; harmless because decide() requires perm.complete, but it breaks the "stderr empty" expectation in that corner. Owner det-stage5, cosmetic.
4. encoder.py:205 (smoke ORT child) and stage5_encoder docstring line 8 do not mention the hash check; encoder.py:416-417 prints the directory size labelled "external data if any" (still cosmetic, carried over from Part A item 3).
5. zone_groups (stage4_probe.py:286-291): a clip with zone_fallback has zone NaN and is counted `mismatch`, not `no_r1` (agree False). Only affects the ablation subset composition, as intended-by-effect but undocumented. Owner det-stage5 (doc note).
6. Inner-selection optimism (disclosed in the protocol): inner scores for j come from models trained with other inner clips' labels incl. j's neighbours; confirmed only inner key is affected (perturbation test).
No leaks of the outer clip's label into its own prediction were found.

Open risks: methodology (n=17, 5 positives, adaptive after stage 4) is the critical-reviewer's scope; the NO-GO result is consistent with the code path, no code defect that would have flipped it was found (permutations full nested, criteria evaluated per protocol).
