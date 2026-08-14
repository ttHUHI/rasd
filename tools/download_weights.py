# -*- coding: utf-8 -*-
"""
download_weights.py — 下载 WRN50 ImageNet 预训练权重（纯 Python，无第三方依赖）
===============================================================================
解决 download.pytorch.org 在部分地区下载不稳定/被截断的问题，
改为从 HuggingFace 镜像（hf-mirror.com）下载。

用法:
  python download_weights.py                      # 下载到 weights/wrn50_imagenet.bin
  python download_weights.py --out my/weights.bin  # 自定义路径

下载后训练时加 --backbone-weights <路径> 即可：
  python train.py --data-root data/MPDD --backbone-weights weights/wrn50_imagenet.bin
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

URL = ("https://hf-mirror.com/nateraw/wide_resnet50_2/resolve/main/pytorch_model.bin")
EXPECTED_BYTES = 275916719  # 已核实的 HF 实际文件大小（约 263MiB）
CHUNK = 1 << 20  # 1MB


def download(url: str, dst: Path):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    # 断点续传：已有部分则从断点继续
    start = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={start}-",
    })
    mode = "ab" if start else "wb"
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, mode) as f:
        total = start + int(r.headers.get("Content-Length", 0) or 0)
        done = start
        while True:
            buf = r.read(CHUNK)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            pct = done / EXPECTED_BYTES * 100
            sys.stdout.write(f"\r{done/1e6:.1f}/{EXPECTED_BYTES/1e6:.1f} MB ({pct:.0f}%)")
            sys.stdout.flush()
    print()
    if done < EXPECTED_BYTES:
        print(f"[警告] 文件不完整: {done} < {EXPECTED_BYTES}，请重跑本脚本续传")
        sys.exit(1)
    tmp.replace(dst)
    print(f"[OK] 权重已保存: {dst} ({done/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weights/wrn50_imagenet.bin")
    args = ap.parse_args()
    download(URL, Path(args.out))


if __name__ == "__main__":
    main()
