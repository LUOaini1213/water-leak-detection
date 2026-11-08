"""Download the BattLeDIM 2020 dataset files used by this project and verify them.

Source: Zenodo record 4017659 (DOI 10.5281/zenodo.4017659), licence CC BY 4.0.

Each file is checked against the MD5 published by Zenodo and against the SHA-256
recorded in ``data/manifest.json`` (committed). Run ``--record`` once to write the
SHA-256 values into the manifest after a first verified download.

Usage:
    python scripts/fetch_data.py            # download missing files, verify all
    python scripts/fetch_data.py --verify   # verify only, no network
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"
RECORD_API = "https://zenodo.org/api/records/4017659/files/{name}/content"


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def download(name: str, dest: Path) -> None:
    url = RECORD_API.format(name=name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {name} ...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def verify(name: str, meta: dict) -> list[str]:
    path = RAW / name
    problems = []
    if not path.exists():
        return [f"{name}: missing"]
    if path.stat().st_size != meta["size"]:
        problems.append(f"{name}: size {path.stat().st_size} != {meta['size']}")
    if file_hash(path, "md5") != meta["md5"]:
        problems.append(f"{name}: md5 mismatch")
    if meta.get("sha256") and file_hash(path, "sha256") != meta["sha256"]:
        problems.append(f"{name}: sha256 mismatch")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="verify only, no download")
    ap.add_argument("--record", action="store_true", help="write sha256 into manifest")
    args = ap.parse_args()

    manifest = load_manifest()
    RAW.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    for name, meta in manifest["files"].items():
        path = RAW / name
        if not args.verify and (not path.exists() or path.stat().st_size != meta["size"]):
            download(name, path)
        problems += verify(name, meta)
        if args.record and path.exists() and not problems:
            meta["sha256"] = file_hash(path, "sha256")
    if problems:
        print("VERIFY FAILED:\n  " + "\n  ".join(problems))
        return 1
    if args.record:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("sha256 recorded in data/manifest.json")
    print(f"OK: {len(manifest['files'])} files verified in data/raw")
    return 0


if __name__ == "__main__":
    sys.exit(main())
