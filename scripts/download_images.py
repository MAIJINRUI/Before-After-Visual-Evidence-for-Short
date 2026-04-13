#!/usr/bin/env python3
"""
download_images.py
从 ALFRED 原始数据中提取 before/after 图片到 data/images/
只下载 image_manifest.json 中记录的图片。
"""

import json, shutil, pathlib

ALFRED_ROOT   = pathlib.Path("json_2.1.0")
MANIFEST_FILE = pathlib.Path("data/image_manifest.json")
SAMPLES_FILE  = pathlib.Path("data/samples.jsonl")
IMG_DIR       = pathlib.Path("data/images")


def main():
    IMG_DIR.mkdir(exist_ok=True)

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    copied = 0
    missing = 0

    for entry in manifest:
        sid = int(entry["sample_id"][1:])  # S0001 → 1
        trial_path = ALFRED_ROOT / entry["trial"]

        for tag, img_name in [("before", entry["before_image_name"]),
                               ("after",  entry["after_image_name"])]:
            if not img_name:
                missing += 1
                continue

            dst = IMG_DIR / f"{sid:04d}_{tag}.png"
            if dst.is_file():
                continue  # already exists

            # Try raw_images / high_res_images
            for subdir in ["raw_images", "high_res_images"]:
                src = trial_path / subdir / img_name
                if src.is_file():
                    shutil.copy2(src, dst)
                    copied += 1
                    break
            else:
                missing += 1

    # Update samples.jsonl with image paths
    if copied > 0:
        samples = []
        with open(SAMPLES_FILE) as f:
            for line in f:
                samples.append(json.loads(line))

        for s in samples:
            sid = int(s["sample_id"][1:])
            b = IMG_DIR / f"{sid:04d}_before.png"
            a = IMG_DIR / f"{sid:04d}_after.png"
            if b.is_file():
                s["before_image"] = f"images/{sid:04d}_before.png"
            if a.is_file():
                s["after_image"] = f"images/{sid:04d}_after.png"

        with open(SAMPLES_FILE, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Copied: {copied} images")
    print(f"⚠️  Missing: {missing} images")

    if missing > 0:
        print(f"\n需要下载 ALFRED 完整数据（含图片）：")
        print(f"  Option 1: 下载 full_2.1.0（含 raw_images）")
        print(f"  Option 2: 用 AI2-THOR 重新渲染")
        print(f"  详见 https://github.com/askforalfred/alfred")


if __name__ == "__main__":
    main()