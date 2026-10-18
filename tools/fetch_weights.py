"""Download YOLOX ONNX person-detector weights into ./weights/ (gitignore it) and verify SHA256.

Source: https://github.com/Megvii-BaseDetection/YOLOX (Apache-2.0), release 0.1.1rc0.
usage: python tools/fetch_weights.py [yolox_s|yolox_tiny|yolox_nano|all|<research snapshot>] [--force]
Exit code != 0 on any download / size / hash mismatch; a bad file is never left in place.

RESEARCH snapshots (never used by the production API; opt-in by name, not part of "all"): pinned
Hugging Face revision, every file with exact size + SHA256 (LFS sha256 from the HF API, small files
hashed from the same pinned revision), stored under weights/<name>/.
  qwen2_vl_2b_instruct_mlx4: mlx-community/Qwen2-VL-2B-Instruct-4bit @ 01af461c (4-bit MLX conversion of
    Qwen/Qwen2-VL-2B-Instruct @ 895c3a49, whose LICENSE file is Apache-2.0; the conversion's card
    declares apache-2.0 and ships no LICENSE file of its own).
  siglip2_base_patch16_224: google/siglip2-base-patch16-224 @ 75de2d55 (stage 5 frozen image encoder;
    card YAML `license: apache-2.0`, no LICENSE file in the repo; code google-research/big_vision @ 0127fb6b
    LICENSE Apache-2.0; training data WebLI, unreleased -> training-data terms UNVERIFIED).
"""
import hashlib
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.detector import DEFAULT_MODEL, MODELS, WEIGHTS_DIR  # noqa: E402


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _ssl_ctx():
    """System CA store; falls back to certifi's bundle (python.org macOS builds ship no CAs)."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


HF_SNAPSHOTS = {
    "siglip2_base_patch16_224": {
        "repo": "google/siglip2-base-patch16-224", "rev": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "files": {
            "README.md": (3375, "39ac3705d62af9ffa1a14675b8ccb220a75f2d81acd530e564a3b1e3dfe418d8"),
            "config.json": (253, "fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7"),
            "model.safetensors": (1500800904, "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"),
            "preprocessor_config.json": (394, "9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc"),
            "special_tokens_map.json": (636, "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351"),
            "tokenizer.json": (34363039, "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"),
            "tokenizer.model": (4241003, "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"),
            "tokenizer_config.json": (47164, "14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8"),
        }},
    "qwen2_vl_2b_instruct_mlx4": {
        "repo": "mlx-community/Qwen2-VL-2B-Instruct-4bit", "rev": "01af461cdb9574acc09084a0ef94e216e142b085",
        "files": {
            "README.md": (565, "14ac3b323646364cdc0fb976e6395933bd6ad89343ff055ef65d97a66154957a"),
            "added_tokens.json": (392, "7fa54985b58718a8fdb4f4d97484c4bd908db114847675e4bf3afe3e1d5d7bd4"),
            "chat_template.json": (1049, "94174d7176c52a7192f96fc34eb2cf23c7c2059d63cdbfadca1586ba89731fb7"),
            "config.json": (1408, "3c6bdaacede67a811e0e122def203d9465301416a3df11835a2f238e1f349b9e"),
            "merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
            "model.safetensors": (1245868907, "3534d3dcfaacd0cd00e935469bc34b6d692b2ba0749c5d9ee422f3f2175cf992"),
            "model.safetensors.index.json": (107605, "ddc9492d4ec6dc9031d4f96050be9a233a451a3a0c07720fbbffc0b950e69669"),
            "preprocessor_config.json": (499, "56fd296c1d6dc1ae216c84b39b10381784a02ff42db425dbf02683b42a7e3f65"),
            "special_tokens_map.json": (613, "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"),
            "tokenizer.json": (11420371, "091aa7594dc2fcfbfa06b9e3c22a5f0562ac14f30375c13af7309407a0e67b8a"),
            "tokenizer_config.json": (4297, "eb3c5a310899d7d65feec0dee4dff9067d7bc4ad2a17e2005bc9163f1ace0531"),
            "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
        }},
}


def _download(url, dst, size, sha256):
    if os.path.isfile(dst) and os.path.getsize(dst) == size and _sha(dst) == sha256:
        return False
    tmp = dst + ".part"
    try:
        with urllib.request.urlopen(url, timeout=60, context=_ssl_ctx()) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                if f.tell() > size:
                    raise ValueError("download larger than expected")
        if os.path.getsize(tmp) != size:
            raise ValueError(f"size {os.path.getsize(tmp)} != {size}")
        if _sha(tmp) != sha256:
            raise ValueError("sha256 mismatch")
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return True


def fetch_snapshot(name: str) -> str:
    m = HF_SNAPSHOTS[name]
    d = os.path.join(WEIGHTS_DIR, name)
    os.makedirs(d, exist_ok=True)
    got = 0
    for fn, (size, sha) in m["files"].items():
        got += _download(f"https://huggingface.co/{m['repo']}/resolve/{m['rev']}/{fn}", os.path.join(d, fn), size, sha)
    print(f"{name}: {len(m['files'])} files sha256 ok ({got} downloaded) -> weights/{name}/")
    return d


def fetch(name: str, force: bool = False) -> str:
    m = MODELS[name]
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    dst = os.path.join(WEIGHTS_DIR, m["file"])
    if os.path.isfile(dst) and not force and _sha(dst) == m["sha256"]:
        print(f"{name}: present, sha256 ok ({m['sha256'][:12]})")
        return dst
    tmp = dst + ".part"
    try:
        with urllib.request.urlopen(m["url"], timeout=60, context=_ssl_ctx()) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                if f.tell() > m["size"]:
                    raise ValueError("download larger than expected")
        if os.path.getsize(tmp) != m["size"]:
            raise ValueError(f"size {os.path.getsize(tmp)} != {m['size']}")
        digest = _sha(tmp)
        if digest != m["sha256"]:
            raise ValueError(f"sha256 mismatch: {digest}")
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"{name}: downloaded, sha256 ok ({m['sha256'][:12]}) -> {dst}")
    return dst


def main(argv):
    force = "--force" in argv
    names = [a for a in argv if not a.startswith("--")] or [DEFAULT_MODEL]
    if names == ["all"]:
        names = list(MODELS)
    bad = [n for n in names if n not in MODELS and n not in HF_SNAPSHOTS]
    if bad:
        print(f"unknown model(s) {bad}; choose from {list(MODELS)} or 'all'", file=sys.stderr)
        return 2
    try:
        for n in names:
            fetch_snapshot(n) if n in HF_SNAPSHOTS else fetch(n, force)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
