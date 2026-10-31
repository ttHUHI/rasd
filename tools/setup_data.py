# -*- coding: utf-8 -*-
"""MPDD 数据集检测/校验/获取指引

用法:
    python tools/setup_data.py                # 检测默认位置(../rsad/data/MPDD)并校验
    python tools/setup_data.py --path D:/data/MPDD   # 指定数据目录
    python tools/setup_data.py --clone        # 数据缺失时自动从官方 GitHub 克隆

说明: 数据集不随代码仓库分发(1.8GB), 获取途径:
  1) 官方 GitHub:   https://github.com/stepanje/MPDD
  2) 本仓库种子:    MPDD.torrent (BitTorrent 下载)
  3) 中文镜像:      https://orion.hyper.ai/datasets/31541
放置后运行本脚本校验通过即可训练。
"""
import argparse
import subprocess
import sys
from pathlib import Path

# 论文 Table 1: 各类训练/测试数量 (train, test)
EXPECTED = {
    "bracket_black": (289, 79),
    "bracket_brown": (185, 77),
    "bracket_white": (110, 60),
    "connector": (128, 44),
    "metal_plate": (54, 97),
    "tubes": (122, 101),
}
OFFICIAL_REPO = "https://github.com/stepanje/MPDD.git"


def check(root):
    root = Path(root)
    ok = True
    print(f"检查数据目录: {root}")
    if not root.exists():
        print("  [缺失] 目录不存在")
        return False
    total_tr = total_te = 0
    for cls, (tr, te) in EXPECTED.items():
        gdir = root / "train" / cls / "good"
        a = len(list(gdir.glob("*.png"))) if gdir.exists() else 0
        tdir = root / "test" / cls
        b = len([f for f in tdir.rglob("*.png")
                 if "mask" not in f.stem.lower()]) if tdir.exists() else 0
        total_tr += a
        total_te += b
        mark = "OK" if (a == tr and b == te) else f"MISMATCH (期望 {tr}/{te})"
        if mark != "OK":
            ok = False
        print(f"  {cls:>14}: train={a:>3} test={b:>3}  {mark}")
    print(f"  合计: train={total_tr} (期望 888) test={total_te} (期望 458)")
    if total_tr != 888 or total_te != 458:
        ok = False
    print("结果:", "通过" if ok else "未通过 (请核对数据来源)")
    return ok


def main():
    ap = argparse.ArgumentParser(description="MPDD 数据检测/校验/获取")
    ap.add_argument("--path", default=None,
                    help="数据目录（默认: 相对项目根的 ../rsad/data/MPDD）")
    ap.add_argument("--clone", action="store_true",
                    help="数据缺失时自动从官方 GitHub 克隆")
    args = ap.parse_args()

    root = Path(args.path) if args.path else \
        Path(__file__).resolve().parent.parent.parent / "rsad" / "data" / "MPDD"

    if not root.exists() or not (root / "train").exists():
        print(f"[提示] 数据不在 {root}")
        print("获取途径:")
        print("  1) 官方 GitHub: https://github.com/stepanje/MPDD")
        print("  2) 本仓库种子:  MPDD.torrent (需 BitTorrent 客户端)")
        print("  3) 中文镜像:    https://orion.hyper.ai/datasets/31541")
        if args.clone:
            print(f"  正在从官方仓库克隆: {OFFICIAL_REPO}")
            root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1", OFFICIAL_REPO, str(root)],
                           check=True)
            print("克隆完成")
        else:
            print("放置数据后重新运行本脚本校验（或用 --clone 自动克隆）")
            return 1
    return 0 if check(root) else 1


if __name__ == "__main__":
    sys.exit(main())
