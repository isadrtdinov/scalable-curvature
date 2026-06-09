# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone curvature evaluation across a directory of saved checkpoints.

For every `*.ckpt` in `--ckpt_dir`, this script:
  1. Loads the checkpoint's saved state into a single model + optimizer instance.
  2. Samples a single evaluation batch of size `--batch_size`.
  3. Computes critical learning rate, preconditioned sharpness, and bare sharpness on
     that batch, splitting the batch into `--num_microbatches` slices so the
     Hessian-vector product (and the preconditioner gradient, and the critical-LR
     forward/backward passes) fit in GPU memory.
  4. Appends a row to a CSV.

Assumption: all checkpoints in `--ckpt_dir` were produced by the same training run
(identical model architecture). The model + optimizer are built once from the first
checkpoint's saved config and reused — this avoids paying torch.compile startup cost
per checkpoint.

The CSV is flushed after every checkpoint so partial progress survives crashes.

Typical invocation:

    python eval_curvature_ckpts.py \
        --ckpt_dir ./checkpoints \
        --dataset_name fineweb \
        --batch_size 64 --num_microbatches 4 \
        --output_csv curvature.csv
"""

import argparse
import glob
import os
from contextlib import nullcontext

import torch
import torch.nn.utils

import utils.sharpness_cupy_utils as sharpness_utils
from utils.critical_learning_rate import compute_critical_learning_rate
from utils.data_loader import get_batch
from utils.data_storage import DataStorageGeneral as DataStorage
from utils.gpt import GPT, GPTConfig
from utils.loss_functions import CrossEntropyLoss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def strip_compile_prefix(state_dict):
    """torch.compile prepends '_orig_mod.' to module names; strip it for load_state_dict."""
    unwanted = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted):
            state_dict[k[len(unwanted):]] = state_dict.pop(k)
    return state_dict


def _get(saved_cfg, name, default=None):
    """Read a field from the saved argparse Namespace, falling back to a default."""
    return getattr(saved_cfg, name, default)


def build_model_and_optim(saved_cfg, device):
    """Reconstruct model + optimizer from the cfg dumped at training time, on `device`."""
    use_bias = _get(saved_cfg, "use_bias")
    if use_bias is None:
        use_bias = str(_get(saved_cfg, "bias", "False")) == "True"

    gpt_conf = GPTConfig(
        context_len=saved_cfg.context_len,
        vocab_size=saved_cfg.vocab_size,
        num_layers=saved_cfg.num_layers,
        num_heads=saved_cfg.num_heads,
        embd_dim=saved_cfg.embd_dim,
        bias=use_bias,
        init_var=_get(saved_cfg, "init_var", 1.0),
        use_flash=_get(saved_cfg, "use_flash", False),
    )
    model = GPT(gpt_conf)

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    optim = model.configure_optimizers(
        learning_rate=_get(saved_cfg, "lr_init", 0.0),
        betas=(saved_cfg.beta1, saved_cfg.beta2),
        eps=saved_cfg.eps,
        weight_decay=saved_cfg.weight_decay,
        device_type=device_type,
    )
    return model, optim, gpt_conf


def discover_checkpoints(ckpt_dir):
    """Return [(step, path)] sorted by step. Loads each ckpt to CPU once to read its step."""
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not paths:
        raise FileNotFoundError(f"No *.ckpt files in {ckpt_dir}")
    out = []
    for p in paths:
        meta = torch.load(p, map_location="cpu", weights_only=False)
        out.append((int(meta["step"]), p))
        del meta
    out.sort(key=lambda t: t[0])
    return out


def load_ckpt_into(model, optim, ckpt_path, device):
    """Load weights + optimizer state from ckpt_path into existing model/optim, then ensure
    the optimizer state tensors live on `device`. Returns the saved config (Namespace)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = strip_compile_prefix(ckpt["model"])
    model.load_state_dict(state_dict)
    optim.load_state_dict(ckpt["optim"])
    for state in optim.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    saved_cfg = ckpt["config"]
    del ckpt, state_dict
    return saved_cfg


def evaluate_one_ckpt(
    model,
    optim,
    cfg,
    saved_cfg,
    device,
    ctx,
    loss_fn,
    eigvec,
    pre_eigvec,
    lr_guess,
):
    """Compute critical-LR + pre-sharpness + sharpness for the current model state. Returns
    a row dict plus updated warm-start state (eigvec, pre_eigvec, lr_guess)."""
    data_dir = os.path.join(cfg.data_root, cfg.dataset_name)
    context_len = saved_cfg.context_len
    X, Y = get_batch(data_dir, "train", context_len, cfg.batch_size, device)

    (lr_lower, lr_upper), num_iters_critical = compute_critical_learning_rate(
        ctx=ctx,
        model=model,
        loss_fn=loss_fn,
        optim=optim,
        batch=(X, Y),
        lr_guess=lr_guess,
        tol_power=cfg.crit_lr_tol_power,
        recompute_grads=True,
        num_microbatches=cfg.num_microbatches,
    )
    optim.zero_grad(set_to_none=True)
    lr_guess_next = lr_lower if lr_lower > 0 else lr_guess

    pre_sharpness, pre_eigvec_next, num_iters_pre_sharpness = (
        sharpness_utils.get_pre_sharpness_lobpcg(
            model,
            loss_fn,
            optim,
            (X, Y),
            eigvecs=pre_eigvec,
            tol=cfg.sharpness_tol,
            max_iters=cfg.max_iters,
            num_microbatches=cfg.num_microbatches,
        )
    )
    optim.zero_grad(set_to_none=True)

    sharpness, eigvec_next, num_iters_sharpness = sharpness_utils.get_sharpness_lobpcg(
        model,
        loss_fn,
        (X, Y),
        eigvecs=eigvec,
        tol=cfg.sharpness_tol,
        max_iters=cfg.max_iters,
        num_microbatches=cfg.num_microbatches,
    )
    optim.zero_grad(set_to_none=True)

    row = dict(
        lr_lower=lr_lower,
        lr_upper=lr_upper,
        num_iters_critical=num_iters_critical,
        pre_sharpness=pre_sharpness,
        pre_num_iters=num_iters_pre_sharpness,
        sharpness=sharpness,
        num_iters=num_iters_sharpness,
    )
    del X, Y
    return row, eigvec_next, pre_eigvec_next, lr_guess_next


def main(cfg, device):
    assert cfg.batch_size % cfg.num_microbatches == 0, (
        f"--batch_size ({cfg.batch_size}) must be divisible by --num_microbatches ({cfg.num_microbatches})"
    )

    os.makedirs(cfg.results_dir, exist_ok=True)
    output_path = os.path.join(cfg.results_dir, cfg.output_csv)

    if "cuda" in str(device) and torch.cuda.is_bf16_supported():
        ptdtype = torch.bfloat16
    elif "cuda" in str(device):
        ptdtype = torch.float16
    else:
        ptdtype = torch.float32
    ctx = (
        nullcontext()
        if "cuda" not in str(device)
        else torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    )

    loss_fn = CrossEntropyLoss()

    ckpts = discover_checkpoints(cfg.ckpt_dir)
    print(f"Found {len(ckpts)} checkpoints in {cfg.ckpt_dir}")

    # Build model + optimizer once from the first checkpoint's config; reuse across all ckpts.
    _, first_path = ckpts[0]
    first_meta = torch.load(first_path, map_location="cpu", weights_only=False)
    template_cfg = first_meta["config"]
    del first_meta

    model, optim, _ = build_model_and_optim(template_cfg, device)
    num_params_full, embd_params = model.get_num_params()
    model.to(device)
    if cfg.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    results = DataStorage(
        columns=[
            "step",
            "num_params",
            "embd_params",
            "batch_size",
            "num_microbatches",
            "lr_lower",
            "lr_upper",
            "num_iters_critical",
            "pre_sharpness",
            "pre_num_iters",
            "sharpness",
            "num_iters",
        ]
    )

    eigvec = None
    pre_eigvec = None
    lr_guess = cfg.lr_guess

    for saved_step, path in ckpts:
        print(f"--- Evaluating step {saved_step} ({os.path.basename(path)}) ---")
        saved_cfg = load_ckpt_into(model, optim, path, device)
        row, eigvec, pre_eigvec, lr_guess = evaluate_one_ckpt(
            model=model,
            optim=optim,
            cfg=cfg,
            saved_cfg=saved_cfg,
            device=device,
            ctx=ctx,
            loss_fn=loss_fn,
            eigvec=eigvec,
            pre_eigvec=pre_eigvec,
            lr_guess=lr_guess,
        )
        row = dict(
            step=saved_step,
            num_params=num_params_full,
            embd_params=embd_params,
            batch_size=cfg.batch_size,
            num_microbatches=cfg.num_microbatches,
            **row,
        )
        print(
            f"step={row['step']}  "
            f"lr=[{row['lr_lower']:.2e},{row['lr_upper']:.2e}]  "
            f"pre_sharpness={row['pre_sharpness']:.4e}  "
            f"sharpness={row['sharpness']:.4e}"
        )
        results.add_entry(**row)
        results.save_to_csv(output_path)
        torch.cuda.empty_cache()

    print(f"Wrote {output_path}")


### CONFIG ###

parser = argparse.ArgumentParser(description="Evaluate curvature across a directory of checkpoints")
parser.add_argument("--ckpt_dir", type=str, required=True,
                    help="directory containing *.ckpt files to evaluate")
parser.add_argument("--dataset_name", type=str, default="fineweb",
                    help="dataset name; data is read from <data_root>/<dataset_name>/{train,val}.bin")
parser.add_argument("--data_root", type=str, default="data",
                    help="directory containing per-dataset subdirectories")
parser.add_argument("--batch_size", type=int, default=64,
                    help="total evaluation batch size (the effective batch for sharpness / critical LR)")
parser.add_argument("--num_microbatches", type=int, default=1,
                    help="split the batch into this many micro-batches for HVP / grad / forward "
                         "passes. Must divide --batch_size.")
parser.add_argument("--results_dir", type=str, default="curvature_results")
parser.add_argument("--output_csv", type=str, default="curvature.csv",
                    help="filename inside --results_dir")
# critical learning rate
parser.add_argument("--crit_lr_tol_power", type=int, default=4)
parser.add_argument("--lr_guess", type=float, default=1e-04,
                    help="initial guess for critical LR on the first checkpoint; subsequent "
                         "checkpoints warm-start from the previous lr_lower")
# sharpness
parser.add_argument("--max_iters", type=int, default=500)
parser.add_argument("--sharpness_tol", type=float, default=1e-09)
# runtime
parser.add_argument("--compile", type=lambda x: x.lower() == "true", default=True)


cfg = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

main(cfg, device)
