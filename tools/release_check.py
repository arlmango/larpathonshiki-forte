"""Read-only release audit for a future code-only publication.

Usage:  python -m tools.release_check [--print-add] [--with-coordination]

Builds an explicit ALLOWLIST of publishable paths, verifies that no DENYLIST
class (local_private/, weights/, data/, .venv*, .claude-flow/, models, video,
jsonl, .env*) is inside it, and scans allowlisted files for secrets / PII-like
strings. It modifies nothing and never executes git (except the read-only
`git check-ignore` / `git ls-files`). It prints only counts for denied trees
and only code locations (file:line) for scan hits; matched text is never
printed (redacted as <hash> / <redacted>).
Exit code: 0 = READY, 1 = NOT READY.
"""
import ast
import fnmatch
import hashlib
import os
import re
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF = "tools/release_check.py"

# ---- allowlist (relative posix globs; fnmatch, '*' also crosses '/') ----
ALLOW = [
    "app/*.py", "app/**/*.py", "app/*.json",
    "tools/*.py",                      # includes _selftest*.py (tests of the tools)
    "tests/*.py", "tests/**/*.py",
    "static/*", "static/**/*",
    "docs/*.md",                       # minus REVIEW_FIRST below
    "README.md", "requirements*.txt", ".gitignore",
]
# Allowlisted by pattern but held back: internal process log (aggregates only,
# still must be read by a human before publishing). Added with --with-coordination.
REVIEW_FIRST = ["docs/COORDINATION.md"]
# Deliberately excluded from publication.
EXCLUDE = [".claude/*", "NEXT_DETECTION_TASK.txt"]

# ---- denylist ----
DENY_DIRS = ["local_private", "weights", "data", ".venv*", ".claude-flow"]
DENY_GLOBS = ["*.onnx", "*.safetensors", "*.mov", "*.mp4", "*.jsonl", ".env*"]
PRUNE = {".git", "__pycache__", ".claude", ".pytest_cache", "node_modules"}
# paths git must ignore (probe names; existence not required)
GITIGNORE_PROBES = [
    "local_private/x", "weights/x", "data/x", ".venv/x", ".venv-research/x",
    ".claude-flow/x", "a.onnx", "a.safetensors", "a.mov", "a.mp4", "a.jsonl", ".env",
    ".env.local",
]

SCAN_EXT = (".py", ".md", ".txt", ".html", ".json")

RX = {
    "api_key_token": re.compile(
        r"\bsk-[A-Za-z0-9_-]{20,}|\bAKIA[0-9A-Z]{16}\b|\bhf_[A-Za-z0-9]{30,}"
        r"|\bghp_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}"
        r"|[Bb]earer\s+[A-Za-z0-9._~+/-]{20,}"),
    "api_key_literal": re.compile(
        r"""API_KEY\s*[=:]\s*['"]?(?!<|\$|['"]?\s*$|\{|os\.|''|"")([A-Za-z0-9+/_-]{8,})"""),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "abs_user_path": re.compile(r"/Users/[A-Za-z0-9._-]+|/home/[a-z_][a-z0-9_-]*|[A-Z]:\\Users\\"),
    "hex16_plus": re.compile(r"\b[0-9a-fA-F]{16,}\b"),
}
HEX_OK_CTX = re.compile(r"sha-?256|shasum|checksum|revision|digest|licen[sc]e|weights?\b|\.onnx|\.safetensors|"
                        r"commit|snapshot", re.I)
HEX_OK_FILES = {"app/detector.py", "tools/fetch_weights.py", "docs/LICENSES.md", "docs/DEPLOY.md"}
CYR = re.compile(r"[А-Яа-яЁё]")
CYR_MIN = 30           # chars of a single non-docstring string literal
BIND_BAD = re.compile(r"0\.0\.0\.0")
BIND_OK = re.compile(r"127\.0\.0\.1")
# only code that turns the flag ON by default: environ.setdefault(.., "1"), environ.get(.., "1"),
# environ[..] = "1"; docstrings / messages that merely mention "FLAG=1" are not hits.
DEFAULT_ON = re.compile(
    r"""environ(\.setdefault\(|\.get\(|\[)\s*["'](ALLOW_UNVALIDATED_RULES|SYNTHETIC_DETECTOR)["']\s*(\]\s*=|,)\s*["']1["']""")
FAKE_USERS = {"a", "x", "u", "user", "name", "me", "foo", "bar", "test", "someone"}
CYR_EXEMPT = {"app/norms.py"}  # public normative text (docs/NORMS.md source); verify once by eye

# ---- KNOWN_OK: the 35 findings reviewed by the user (release-findings-review) ----
# Each entry downgrades exactly one location: (kind, file, sha256[:16], reason).
#   hex: sha256 of the stripped LINE text;  cyr: sha256 of the string LITERAL value.
# Matching is by file + hash, never by class; any other hex/Cyrillic is still reported.
# The hashes stand in for the text, so the checker itself holds no scene/label content.
R_SUM = "sha256/revision constant next to a checksum/revision label (not an access key)"
R_JOB = "synthetic job-id example in a request sample (not a real job)"
R_DEMO = "public Russian caption/warning/aggregate literal of the demo video (no annotator answers, no scene text)"
R_TPL = "static HTML/JS template of the local annotation UI (data is substituted at run time, none in the literal)"
R_PROV = "standard data-provenance note and its test"
KNOWN_OK = [
    ("hex", "docs/PROTOCOL_STAGE5.md", "8e7197bb9573030d", R_SUM),
    ("hex", "docs/PROTOCOL_STAGE5.md", "73eded9c9a45ff9e", R_SUM),
    ("hex", "docs/REVIEW_CRITICAL.md", "ccd20d1214453c87", R_SUM),
    ("hex", "docs/REVIEW_CRITICAL.md", "7a3f00ef82cf6741", R_SUM),
    ("hex", "docs/REVIEW_CRITICAL.md", "2ae0e808563e2ba3", R_SUM),
    ("hex", "docs/SUBMISSION.md", "a2cb081fc25d1306", R_JOB),
    ("hex", "tools/stage4_probe.py", "dd9fd82746021b64", R_SUM),
    ("hex", "tools/stage4_probe.py", "29dd6ceadad90f7f", R_SUM),
    ("hex", "tools/stage4_probe.py", "d6fb56d6d5d09f75", R_SUM),
    ("cyr", "tests/test_evaluate.py", "32ccb3b47d97bb0d", R_PROV),
    ("cyr", "tools/evaluate.py", "32ccb3b47d97bb0d", R_PROV),
    ("cyr", "tools/annotate.py", "5e217680b3bde43e", R_TPL),
    ("cyr", "tools/annotate_attrs.py", "05e0aa34e596ff9b", R_TPL),
    ("cyr", "tools/annotate_audit_ui.py", "c7794f43321291df", R_TPL),
] + [("cyr", "tools/demo_video.py", h, R_DEMO) for h in (
    "492424c054e80075", "3520b8c49c562836", "4c982a0868ce4ebf", "b0f862b7323bf3e0", "a1df3d8af90f1d53",
    "3019c0d65005cf68", "172d539c4a273fde", "ea4e8e49d099791b", "be6eac31e46c7db4", "23a3ff7eb110c45e",
    "b6410d18be4e8ab7", "cd32c0a6bcebbc67", "257260dcae452ef9", "1fc959e648ccb3a9", "614793525991056b",
    "a203e3d866a0fcc6", "a112049219cec4e4", "dd6e21a918207ec7", "eb48e49ec614d201", "0b0e9dc51785ff9c",
    "be4b66f3ba60bded")]
_KNOWN = {(k, f, h) for k, f, h, _ in KNOWN_OK}


def _h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def match_any(r, pats):
    return any(fnmatch.fnmatch(r, p) for p in pats)


def deny_class(r):
    parts = r.split("/")
    for d in DENY_DIRS:
        if any(fnmatch.fnmatch(x, d) for x in parts[:-1]) or (len(parts) == 1 and fnmatch.fnmatch(parts[0], d)):
            return "dir:" + d
    name = parts[-1]
    for g in DENY_GLOBS:
        if fnmatch.fnmatch(name, g):
            return "glob:" + g
    return None


def walk_repo():
    """Return (candidates, deny_counts, unclassified_top). Denied trees are counted, never listed."""
    cands, deny_counts, top_other = [], {}, []
    for cur, dirs, files in os.walk(ROOT):
        r = rel(cur) if cur != ROOT else ""
        keep = []
        for d in dirs:
            if d in PRUNE:
                continue
            dr = (r + "/" + d).lstrip("/")
            cls = deny_class(dr + "/x")
            if cls and cls.startswith("dir:"):
                n = sum(len(f) for _, _, f in os.walk(os.path.join(cur, d)))
                deny_counts[cls] = deny_counts.get(cls, 0) + n
                continue
            keep.append(d)
        dirs[:] = keep
        for f in files:
            fr = (r + "/" + f).lstrip("/")
            cands.append(fr)
    return sorted(cands), deny_counts


def build(with_coord):
    cands, deny_counts = walk_repo()
    allow, review_held, excluded, unclassified, deny_in_tree = [], [], [], [], {}
    for c in cands:
        if c in REVIEW_FIRST:
            (allow if with_coord else review_held).append(c)
            continue
        dc = deny_class(c)
        if dc:
            deny_in_tree[dc] = deny_in_tree.get(dc, 0) + 1   # counted, name not printed
            continue
        if match_any(c, EXCLUDE):
            excluded.append(c)
        elif match_any(c, ALLOW):
            allow.append(c)
        else:
            unclassified.append(c)
    for k, v in deny_in_tree.items():
        deny_counts[k] = deny_counts.get(k, 0) + v
    return sorted(allow), review_held, excluded, unclassified, deny_counts, deny_in_tree


def git_check_ignore(path):
    try:
        r = subprocess.run(["git", "-C", ROOT, "check-ignore", "-q", "--", path],
                           capture_output=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return None


def git_tracked_count():
    try:
        r = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True, timeout=20)
        return len([x for x in r.stdout.splitlines() if x])
    except Exception:
        return -1


def cyrillic_literals(path, text):
    """Lines of non-docstring str constants with >= CYR_MIN Cyrillic chars (py only)."""
    out = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    doc_ids = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = getattr(n, "body", [])
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant):
                doc_ids.add(id(b[0].value))
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_ids:
            if len(CYR.findall(n.value)) >= CYR_MIN:
                out.append((n.lineno, _h(n.value)))
    return sorted(set(out))


def scan(files):
    hits = {k: [] for k in ("api_key_token", "api_key_literal", "private_key_block", "abs_user_path",
                            "hex16_plus", "hex16_known_ctx", "cyrillic_in_code", "bind_0000_in_code",
                            "default_on_flag", "abs_path_test_fixture", "known_ok")}
    info = {"bind_127_in_code": 0, "files_scanned": 0, "self_skipped": 0}
    for f in files:
        if not f.endswith(SCAN_EXT):
            continue
        if f == SELF:
            info["self_skipped"] += 1
            continue
        try:
            text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        except Exception:
            continue
        info["files_scanned"] += 1
        is_code = f.endswith((".py", ".html")) and f.split("/")[0] in ("app", "tools", "static", "tests")
        is_test = f.startswith("tests/")
        for i, line in enumerate(text.splitlines(), 1):
            for k in ("api_key_token", "api_key_literal", "private_key_block", "abs_user_path"):
                m = RX[k].search(line)
                if m:
                    if k == "abs_user_path" and is_test and all(
                            u.lower() in FAKE_USERS for u in re.findall(r"/Users/([A-Za-z0-9._-]+)", line)):
                        hits["abs_path_test_fixture"].append((f, i))
                    else:
                        hits[k].append((f, i))
            if RX["hex16_plus"].search(line):
                known = f in HEX_OK_FILES or HEX_OK_CTX.search(line)
                if known:
                    hits["hex16_known_ctx"].append((f, i))
                elif ("hex", f, _h(line.strip())) in _KNOWN:
                    hits["known_ok"].append((f, i, "hex"))
                else:
                    hits["hex16_plus"].append((f, i))
            if is_code:
                if BIND_BAD.search(line):
                    hits["bind_0000_in_code"].append((f, i))
                if BIND_OK.search(line):
                    info["bind_127_in_code"] += 1
                if f.startswith(("app/", "tools/")) and "selftest" not in f and DEFAULT_ON.search(line):
                    hits["default_on_flag"].append((f, i))
        if f.endswith(".py") and f.split("/")[0] in ("app", "tools", "tests"):
            for ln, hh in ([] if f in CYR_EXEMPT else cyrillic_literals(f, text)):
                if ("cyr", f, hh) in _KNOWN:
                    hits["known_ok"].append((f, ln, "cyr"))
                else:
                    hits["cyrillic_in_code"].append((f, ln))
    return hits, info


def fmt(hl, limit=40):
    s = ", ".join(f"{f}:{i}" for f, i in hl[:limit])
    return s + (f" ... (+{len(hl) - limit})" if len(hl) > limit else "")


def main(argv):
    with_coord = "--with-coordination" in argv
    allow, held, excluded, unclassified, deny_counts, deny_in_tree = build(with_coord)

    if "--print-add" in argv:
        # Printed only; this tool never runs git add.
        print("git add -- " + " \\\n  ".join(shlex.quote(p) for p in allow))
        return 0

    print(f"== release_check (read-only) root={os.path.basename(ROOT)} ==")
    print(f"git ls-files (tracked) : {git_tracked_count()}")
    print(f"ALLOWLIST files        : {len(allow)}")
    by_top = {}
    for p in allow:
        t = p.split("/")[0] if "/" in p else "(root)"
        by_top[t] = by_top.get(t, 0) + 1
    print("  by area              : " + ", ".join(f"{k}={v}" for k, v in sorted(by_top.items())))
    print(f"held for human review  : {len(held)} " + (f"({', '.join(held)}; NOT in --print-add unless --with-coordination)" if held else ""))
    print(f"excluded by decision   : {len(excluded)} (.claude/**, NEXT_DETECTION_TASK.txt)")
    print(f"unclassified in tree   : {len(unclassified)} " + (fmt([(p, 0) for p in unclassified]).replace(":0", "") if unclassified else ""))

    print("\n-- DENYLIST (counts only, contents never listed) --")
    for cls in sorted(set(DENY_DIRS_LABELS()) | set(deny_counts)):
        print(f"  present {cls:<18}: {deny_counts.get(cls, 0)} files")
    n_deny_in_allow = sum(1 for p in allow if deny_class(p))
    print(f"  denylisted paths inside ALLOWLIST: {n_deny_in_allow}")
    print(f"  denylisted files in allowlisted dirs (not allowlisted, name hidden): {sum(deny_in_tree.values())}")

    print("\n-- .gitignore coverage of denylist (git check-ignore on probe names) --")
    gaps = [p for p in GITIGNORE_PROBES if git_check_ignore(p) is False]
    print(f"  probes={len(GITIGNORE_PROBES)} not ignored={len(gaps)}: " + (", ".join(gaps) if gaps else "none"))

    hits, info = scan(allow)
    print(f"\n-- SECRET / PII SCAN ({info['files_scanned']} files; {SELF} skipped from scan: {info['self_skipped']}) --")
    labels = [
        ("api_key_token", "BLOCK", "key/token patterns (sk-, AKIA, hf_, ghp_, Bearer)"),
        ("api_key_literal", "BLOCK", "API_KEY= with a literal value"),
        ("private_key_block", "BLOCK", "private key blocks"),
        ("abs_user_path", "BLOCK", "absolute user paths (/Users/x, /home/x)"),
        ("bind_0000_in_code", "BLOCK", "hard-coded 0.0.0.0 in code (app/tools/static/tests)"),
        ("default_on_flag", "BLOCK", "ALLOW_UNVALIDATED_RULES / SYNTHETIC_DETECTOR defaulting on (app/tools)"),
        ("hex16_plus", "REVIEW", "16+ hex strings outside known hash contexts (<hash>)"),
        ("cyrillic_in_code", "REVIEW", f"Cyrillic string literals >= {CYR_MIN} chars in app/tools/tests (<redacted>)"),
        ("abs_path_test_fixture", "INFO", "fake-user paths (/Users/a|x) used as redaction-test inputs in tests/"),
        ("hex16_known_ctx", "INFO", "16+ hex in sha256/revision/licence contexts (not flagged)"),
    ]
    block = review = 0
    for k, sev, desc in labels:
        n = len(hits[k])
        if sev == "BLOCK":
            block += n
        elif sev == "REVIEW":
            review += n
        print(f"  [{sev:<6}] {k:<18} {n:>4}  {desc}")
        if n and sev != "INFO":
            print(f"             at {fmt([(a[0], a[1]) for a in hits[k]])}")
    kn = hits["known_ok"]
    print(f"  [KNOWN ] suppressed known-OK   {len(kn):>4}  (hex={sum(1 for x in kn if x[2] == 'hex')}, "
          f"cyr={sum(1 for x in kn if x[2] == 'cyr')}; table KNOWN_OK has {len(KNOWN_OK)} entries; "
          f"matched by exact file+hash only)")
    stale = len(KNOWN_OK) - len(kn)
    print(f"             known-OK entries not matched any more (stale/changed): {max(stale, 0)}")
    print(f"  info: 127.0.0.1 mentions in code: {info['bind_127_in_code']}")

    blocking = block + n_deny_in_allow
    review_total = review + len(gaps) + len(unclassified)
    print("\n-- SUMMARY --")
    print(f"  blocking={blocking}  manual-review={review_total} (hex={len(hits['hex16_plus'])}, cyrillic={len(hits['cyrillic_in_code'])}, "
          f"gitignore gaps={len(gaps)}, unclassified={len(unclassified)})  known_ok(suppressed)={len(kn)}")
    tracked = git_tracked_count()
    git_empty = tracked <= 0
    if git_empty:
        print("  git: nothing tracked yet (git ls-files = 0); publication candidate not committed -> NOT READY by rule")
    ready = blocking == 0 and review_total == 0 and not git_empty
    print("RESULT: " + ("READY" if ready else "NOT READY"))
    return 0 if ready else 1


def DENY_DIRS_LABELS():
    return ["dir:" + d for d in DENY_DIRS] + ["glob:" + g for g in DENY_GLOBS]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
