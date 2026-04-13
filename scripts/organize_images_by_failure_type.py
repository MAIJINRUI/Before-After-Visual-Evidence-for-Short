#!/usr/bin/env python3
"""
Create data/images_by_type/{F1..F5}/ with symlinks to before/after images.
Re-run safe: skips existing symlinks with correct target.

Expected files in data/images/ match samples.jsonl, e.g. 0063_before.jpg.
Files named like ._0063_before.jpg are macOS AppleDouble sidecars (not realpixels). Remove them: find data/images -name '._*' -delete
Then re-extract images.zip or run scripts that produce the real JPEGs/PNGs.

Usage (from repo root):
  python scripts/organize_images_by_failure_type.py
  python scripts/organize_images_by_failure_type.py -q
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_real_image_file(path: Path) -> bool:
    """Reject AppleDouble sidecars and other non-image junk (by magic bytes)."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return False
    if len(head) < 3:
        return False
    # JPEG
    if head[:3] == b"\xff\xd8\xff":
        return True
    # PNG
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    # GIF
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return True
    # WebP (RIFF....WEBP)
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return True
    return False


def resolve_image_src(root: Path, rel: str) -> Tuple[Optional[Path], str]:
    """
    Return (absolute path to usable image, reason if missing).
    reason: ok | missing | not_image | appledouble_only | tried_png
    """
    if not rel:
        return None, "missing"
    p = (root / rel).resolve()
    parent = p.parent
    base = p.name
    sidecar = parent / f"._{base}"

    candidates: list[Path] = [p]
    if p.suffix.lower() == ".jpg":
        candidates.append(p.with_suffix(".png"))

    for cand in candidates:
        if not cand.exists() or not cand.is_file():
            continue
        if is_real_image_file(cand):
            if cand != p and p.suffix.lower() == ".jpg":
                return cand, "ok_png_fallback"
            return cand, "ok"
        # exists but not a real image
        return None, "not_image"

    if sidecar.exists():
        return None, "appledouble_only"
    return None, "missing"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Symlink images into per-failure-type folders")
    p.add_argument("--data", type=Path, default=Path("data/samples.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/images_by_type"))
    p.add_argument("--copy", action="store_true", help="copy files instead of symlink")
    p.add_argument("-q", "--quiet", action="store_true", help="only print summary (+ hints)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    data_path = (root / args.data).resolve()
    out_root = (root / args.out).resolve()

    samples = load_jsonl(data_path)
    out_root.mkdir(parents=True, exist_ok=True)
    linked = 0
    skipped_exists = 0
    reasons: Counter[str] = Counter()
    png_fallbacks = 0

    for s in samples:
        ft = s.get("failure_type") or "unknown"
        sid = s.get("sample_id", "?")
        sub = out_root / ft
        sub.mkdir(parents=True, exist_ok=True)

        for role, key in (("before", "before_image"), ("after", "after_image")):
            rel = s.get(key)
            if not rel:
                reasons["missing"] += 1
                continue
            src, why = resolve_image_src(root, rel)
            if why == "ok_png_fallback" and src is not None:
                png_fallbacks += 1
            if src is None:
                reasons[why] += 1
                if not args.quiet:
                    extra = ""
                    if why == "appledouble_only":
                        extra = f" (only macOS ._ sidecar next to expected file; delete ._*)"
                    elif why == "not_image":
                        extra = " (file exists but is not JPEG/PNG — often corrupted or ._ data)"
                    print(f"skip {sid} {key}: {root / rel}{extra}")
                continue

            dest_name = f"{sid}_{role}{src.suffix}"
            dest = sub / dest_name
            if dest.exists() or dest.is_symlink():
                skipped_exists += 1
                continue
            try:
                if args.copy:
                    shutil.copy2(src, dest)
                else:
                    dest.symlink_to(os.path.relpath(src, dest.parent))
                linked += 1
            except OSError as e:
                print(f"fail {dest}: {e}")

    print(
        f"images_by_type: created {linked} links; "
        f"skipped unresolved {sum(reasons.values())}; "
        f"already present {skipped_exists}; "
        f"png_fallback {png_fallbacks}"
    )
    if reasons.get("appledouble_only", 0) or reasons.get("not_image", 0):
        print(
            "\nHint: If you only see files like data/images/._0000_after.jpg, those are NOT images.\n"
            "  find data/images -name '._*' -delete\n"
            "  Then unzip images.zip again (or regenerate JPEGs) so you have e.g. 0000_after.jpg"
        )


if __name__ == "__main__":
    main()
