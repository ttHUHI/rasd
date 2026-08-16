"""YAML 配置加载 + CLI 覆盖"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

# 配置文件里的相对路径字段一律相对项目根目录（rsad_v2 所在目录）解析，
# 与原本"cd rsad_v2 后运行"的语义一致，且从任意 cwd 运行结果不变。
_REL_PATH_KEYS = ("data_root", "backbone_weights")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _merge(dst, src):
    """浅层合并，src 覆盖 dst。"""
    for k, v in vars(src).items():
        setattr(dst, k, v)
    return dst


def _resolve_rel_paths(ns):
    """把相对路径字段解析为相对项目根的绝对路径（规范化 ..）。"""
    for k in _REL_PATH_KEYS:
        v = getattr(ns, k, None)
        if isinstance(v, str) and v and not Path(v).is_absolute():
            setattr(ns, k, str((PROJECT_ROOT / v).resolve()))
    return ns


def load_config(yaml_path=None, cli_args=None):
    """加载一个或多个 YAML，返回合并后的 SimpleNamespace。

    相对路径字段（data_root / backbone_weights）统一按项目根目录解析，
    因此从任意 cwd 运行结果一致；CLI 覆盖值保持原样（相对 cwd 语义）。
    """
    cfg = SimpleNamespace()

    # built-in default
    default_yaml = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
    if default_yaml.exists():
        _merge(cfg, _resolve_rel_paths(_yaml_to_ns(default_yaml)))

    # user-provided yaml (可相对)
    if yaml_path:
        user = Path(yaml_path)
        if not user.is_absolute():
            user = Path.cwd() / user
        if user.exists():
            _merge(cfg, _resolve_rel_paths(_yaml_to_ns(user)))

    # CLI override
    if cli_args is not None:
        _merge(cfg, SimpleNamespace(**{k: v for k, v in vars(cli_args).items() if v is not None}))

    return cfg


def _yaml_to_ns(path):
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return _dotdict_to_ns(d)


def _dotdict_to_ns(d):
    """递归把嵌套 dict 压成 SimpleNamespace，并把嵌套叶子键提升到顶层。

    规则（代码各处按顶层读取配置）：
      - 保留嵌套结构（cfg.model.dim 可用）；
      - 同时把一层嵌套的叶子键复制到顶层（cfg.model.dim → cfg.dim）；
      - 顶层已有同名键时不被嵌套覆盖（顶层优先，CLI 覆盖即走顶层）。
    """
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dotdict_to_ns(v))
        else:
            setattr(ns, k, v)
    # 一层嵌套叶子提升到顶层（如 model.dim → cfg.dim）
    for k, v in list(vars(ns).items()):
        if isinstance(v, SimpleNamespace):
            for sk, sv in vars(v).items():
                if not hasattr(ns, sk):
                    setattr(ns, sk, sv)
    return ns


def build_parser(description="RSAD training"):
    """返回 argparse.ArgumentParser，包含 --config 和常用覆盖项。"""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None, help="YAML 配置文件路径")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-d", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--backbone-weights", default=None)
    ap.add_argument("--cls", default=None,
                    help="per-class 模式: 只训练/评估该类别 (如 metal_plate)")
    ap.add_argument("--feature-norm", action="store_true", default=None,
                    help="对判别器输入做 per-position L2 归一化（实验性, 默认关）")
    ap.add_argument("--noise-smooth", default=None,
                    help="平滑噪声 k<奇数>s<高斯σ>, 如 k5s1.0（生成连续异常区域, 实验性）")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=None)  # for evaluate/predict
    ap.add_argument("--input", default=None)
    ap.add_argument("--vis", default=None)
    ap.add_argument("--th", type=float, default=None,
                    help="判定阈值（默认 0.5; --cls 路由时自动读 <out>/<cls>/threshold.json）")
    return ap
