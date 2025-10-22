# mfm/utils/nan_debug.py
import os
import torch
import torch.nn as nn
from contextlib import suppress

try:
    from mfm.flow_matchers.ema import EMA
except Exception:
    EMA = ()  # 型チェック回避

def _unwrap_ema(m):
    return m.model if isinstance(m, EMA) else m

def _tstats(x: torch.Tensor):
    return dict(
        shape=tuple(x.shape),
        dtype=str(x.dtype),
        device=str(x.device),
        min=float(torch.nanmin(x).detach().cpu()),
        max=float(torch.nanmax(x).detach().cpu()),
        mean=float(torch.nanmean(x).detach().cpu()),
        std=float(torch.nanstd(x).detach().cpu()),
        has_nan=(~torch.isfinite(x)).any().item(),
    )

def _save_dump(dump_dir, tag, payload: dict):
    try:
        os.makedirs(dump_dir, exist_ok=True)
        torch.save(payload, os.path.join(dump_dir, f"{tag}.pt"))
    except Exception as e:
        print(f"[nan_debug] failed to save dump: {e}")

def _check_tensor(name, x, dump_dir, extra=None):
    if not torch.is_tensor(x):
        return
    if torch.isfinite(x).all():
        return
    pay = {"name": name, "stats": _tstats(x)}
    if extra:
        pay.update(extra)
    _save_dump(dump_dir, f"nonfinite_{name}", pay)
    raise RuntimeError(f"[NaN/Inf detected] {name}: {pay['stats']} (dump saved)")

def _make_forward_hook(name, dump_dir):
    def hook(mod, inputs, output):
        # 入力
        for i, a in enumerate(inputs):
            if torch.is_tensor(a):
                _check_tensor(f"{name}.input[{i}]", a, dump_dir)
        # 出力
        if torch.is_tensor(output):
            _check_tensor(f"{name}.output", output, dump_dir)
        elif isinstance(output, (tuple, list)):
            for j, b in enumerate(output):
                if torch.is_tensor(b):
                    _check_tensor(f"{name}.output[{j}]", b, dump_dir)
    return hook

def _make_grad_hook(name, dump_dir):
    def hook(grad):
        _check_tensor(f"{name}.grad", grad, dump_dir)
        return grad
    return hook

def attach_nan_hooks(*, geopath_net: nn.Module, flow_net: nn.Module, working_dir: str, run_tag: str = "run"):
    """数値を変更せず、NaN/Inf を見つけた瞬間に例外＆ダンプするフックを仕込む"""
    dump_dir = os.path.join(working_dir, "debug_dumps", run_tag)
    os.makedirs(dump_dir, exist_ok=True)

    gp_inner = _unwrap_ema(geopath_net)
    fn_inner = _unwrap_ema(flow_net)

    # forward hooks
    gp_inner.register_forward_hook(_make_forward_hook("geopath", dump_dir))
    fn_inner.register_forward_hook(_make_forward_hook("flow", dump_dir))

    # grad hooks（パラメータ毎）
    for n, p in gp_inner.named_parameters(recurse=True):
        if p.requires_grad:
            p.register_hook(_make_grad_hook(f"geopath.{n}", dump_dir))
    for n, p in fn_inner.named_parameters(recurse=True):
        if p.requires_grad:
            p.register_hook(_make_grad_hook(f"flow.{n}", dump_dir))

    # 便利:重要な中間量を明示的にチェックする API（呼び出し側で使う）
    def check_intermediates(**tensors):
        for k, v in tensors.items():
            if torch.is_tensor(v):
                _check_tensor(k, v, dump_dir)
    return check_intermediates