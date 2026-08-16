# -*- coding: utf-8 -*-
"""
mpdd_prepare.py — 解压 MPDD.zip 并重排为 train.py 期望的布局
==================================================================
原始布局（官方 MPDD）:
  MPDD/MPDD/<class>/train/good/*.png
  MPDD/MPDD/<class>/test/good/*.png
  MPDD/MPDD/<class>/test/<defect>/*.png
  MPDD/MPDD/<class>/ground_truth/<defect>/*_mask.png   <- 掩码独立存放

重排后（train.py 期望）:
  data/MPDD/train/<class>/good/*.png
  data/MPDD/test/<class>/good/*.png
  data/MPDD/test/<class>/<defect>/*.png + *_mask.png   <- 掩码与缺陷图同目录

用法:
  python mpdd_prepare.py --zip mpdd/MPDD/data/MPDD.zip --out data/MPDD
"""
import argparse
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="MPDD.zip 路径")
    ap.add_argument("--out", default="data/MPDD", help="输出目录（将包含 train/ 和 test/）")
    ap.add_argument("--verify-only", action="store_true", help="只校验已解压目录，不解压")
    args = ap.parse_args()

    zin = Path(args.zip)
    out = Path(args.out)
    if not args.verify_only:
        assert zin.exists(), f"找不到 zip: {zin}"
        print(f"解压并重排: {zin} -> {out}")

    if args.verify_only:
        verify(out)
        return 0

    with zipfile.ZipFile(zin) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        cnt = Counter()
        wrote = 0
        for n in names:
            parts = n.split("/")
            if len(parts) < 5 or parts[0] != "MPDD" or parts[1] != "MPDD":
                continue
            cls, rest = parts[2], parts[3:]
            # train/good/NNN.png
            if len(rest) >= 3 and rest[0] == "train" and rest[1] == "good":
                dst = out / "train" / cls / "good" / rest[2]
            # test/good/NNN.png
            elif len(rest) >= 3 and rest[0] == "test" and rest[1] == "good":
                dst = out / "test" / cls / "good" / rest[2]
            # test/<defect>/NNN.png
            elif len(rest) >= 3 and rest[0] == "test":
                dst = out / "test" / cls / rest[1] / rest[2]
            # ground_truth/<defect>/NNN_mask.png -> test/<class>/<defect>/NNN_mask.png
            elif len(rest) >= 3 and rest[0] == "ground_truth":
                dst = out / "test" / cls / rest[1] / rest[2]
            else:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
            wrote += 1
            key = "train" if "train" in parts else ("mask" if "mask" in dst.name.lower() else "test")
            cnt[key] += 1
        print(f"写出文件: {wrote}")

    # 校验
    verify(out)
    return 0


def verify(out: Path) -> bool:
    """校验重排后的目录：训练 888 正常 / 测试 176 正常 + 282 缺陷 + 282 掩码。"""
    out = Path(out)
    print("\n=== 校验 ===")
    train_n = test_good = test_def = masks = 0
    ok_all = True
    for cls_dir in sorted((out / "test").iterdir()):
        if not cls_dir.is_dir():
            continue
        g = len(list((cls_dir / "good").glob("*"))) if (cls_dir / "good").exists() else 0
        t = m = 0
        for d in cls_dir.iterdir():
            if d.is_dir() and d.name != "good":
                files = [f for f in d.iterdir() if f.is_file()]
                m += len([f for f in files if "mask" in f.name.lower()])
                t += len([f for f in files if "mask" not in f.name.lower()])
        tr = len(list((out / "train" / cls_dir.name / "good").glob("*"))) if (out / "train" / cls_dir.name / "good").exists() else 0
        train_n += tr
        test_good += g
        test_def += t
        masks += m
        per_ok = (t == m)  # 每个缺陷图应有对应掩码
        ok_all &= per_ok
        print(f"  {cls_dir.name}: train={tr} test_good={g} defect={t} mask={m} "
              f"{'[OK]' if per_ok else '[MASK MISMATCH]'}")
    print(f"\n训练正常: {train_n}（期望 888）| 测试正常: {test_good}（期望 176）| "
          f"测试缺陷: {test_def}（期望 282）| 掩码: {masks}（期望 282）")
    ok = ok_all and train_n == 888 and test_good == 176 and test_def == 282 and masks == 282
    print("==> 校验", "通过 [OK]" if ok else "不匹配 [FAIL]")
    return ok


if __name__ == "__main__":
    sys.exit(main())
