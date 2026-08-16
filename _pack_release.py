# -*- coding: utf-8 -*-
"""打包 rsad_v2 发布版（代码+最终模型），排除 git/缓存/历史实验/备份（临时脚本）"""
import zipfile
from pathlib import Path

SRC = Path(r"D:\RSAD\rsad_v2")
OUT = Path(r"D:\RSAD\rsad_v2_release.zip")

EXCLUDE_DIRS = {
    ".git", "__pycache__",
    # 历史实验 checkpoints（只保留最终 checkpoints_pc80）
    "checkpoints_ab_fc", "checkpoints_ab_fc_layer", "checkpoints_ab_mona",
    "checkpoints_ab_none", "checkpoints_bb_scan", "checkpoints_final",
    "checkpoints_fix", "checkpoints_fix_s01", "checkpoints_fix_s015",
    "checkpoints_pc", "checkpoints_pc30", "checkpoints_pc80_backup",
    "checkpoints_prob", "checkpoints_smoke", "checkpoints_v2_test",
}
EXCLUDE_EXT = {".pyc"}

n_files = 0
total = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(SRC.rglob("*")):
        if f.is_dir():
            continue
        if any(part in EXCLUDE_DIRS for part in f.relative_to(SRC).parts):
            continue
        if f.suffix.lower() in EXCLUDE_EXT:
            continue
        z.write(f, f.relative_to(SRC))
        n_files += 1
        total += f.stat().st_size

print(f"打包完成: {OUT}")
print(f"文件数: {n_files}, 原始大小: {total/1024/1024:.1f} MB")
print(f"zip 大小: {OUT.stat().st_size/1024/1024:.1f} MB")
