# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Full-batch GD debugging driver on Tiny Shakespeare.

Same scientific instrumentation as ``train_gpt_adam_forward_ckpts.py``
(critical-LR + LOBPCG sharpness, preconditioned and raw, via
``utils.sharpness_cupy_utils``) but the "batch" is a fixed set of
``gradient_accumulation_steps * batch_size`` windows sampled once at
init and reused for every training step, the critical-LR search and
both sharpness calls. Single-GPU only; no checkpoint resume.
"""

import argparse
import hashlib
import os
from contextlib import nullcontext

import numpy as np
import torch
import utils.sharpness_cupy_utils as sharpness_utils
from utils.critical_learning_rate import compute_critical_learning_rate
from utils.data_loader import get_batch
from utils.data_storage import DataStorage
from utils.gpt import GPT, GPTConfig
from utils.loss_functions import CrossEntropyLoss
from utils.schedules_utils import warmup_stable_decay

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def estimate_loss(
    ctx, model, loss_fn, eval_steps, data_dir, context_len, batch_size, device
):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            X, Y = get_batch(data_dir, split, context_len, batch_size, device)
            with ctx:
                logits = model(X)
                loss = loss_fn(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_run_filename(cfg, num_steps):
    base_filename = (
        f"{cfg.dataset_name}_"
        f"v{cfg.vocab_size}_"
        f"{cfg.model_name}_"
        f"var{cfg.init_var:0.1f}_"
        f"d{cfg.num_layers}_"
        f"h{cfg.num_heads}_"
        f"n{cfg.embd_dim}_"
        f"c{cfg.context_len}_"
        f"{cfg.optim_name}_"
        f"Tw{cfg.warmup_steps}_"
        f"r{cfg.warmup_exponent}_"
        f"Ts{cfg.stable_steps}_"
        f"{cfg.decay_schedule_name}_"
        f"p{cfg.decay_exponent}_"
        f"T{num_steps}_"
        f"ga{cfg.gradient_accumulation_steps}_"
        f"lr{cfg.lr_peak:.0e}_"
        f"lr{cfg.lr_min_factor}_"
        f"wd{cfg.weight_decay}_"
        f"bs{cfg.batch_size}_"
        f"b{cfg.beta1}_"
        f"b{cfg.beta2}_"
        f"eps{cfg.eps}_"
        f"gc{cfg.grad_clip}_"
        f"mt{cfg.max_train_tokens}_"
        f"ds{cfg.data_seed}"
    )
    return base_filename


def get_base_filename(cfg, num_steps):
    add_filename = (
        f"tol{cfg.crit_lr_tol_power}_"
        f"rg{cfg.recompute_grads}_"
        f"k{cfg.topk}_"
        f"n{cfg.max_iters}_"
        f"tol{cfg.sharpness_tol}"
    )
    return f"{get_run_filename(cfg, num_steps)}_{add_filename}"


def build_full_batch(cfg, device):
    """Sample N = ga*bs windows once and pack them into a list of micro-batches
    plus a concatenated (X_all, Y_all) view for the sharpness/critical-LR calls."""

    data_dir = os.path.join("data", cfg.dataset_name)
    data = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    if cfg.max_train_tokens > 0:
        data = data[: cfg.max_train_tokens]

    assert len(data) > cfg.context_len, (
        f"train.bin has {len(data)} tokens but context_len is {cfg.context_len}"
    )

    g = torch.Generator().manual_seed(cfg.data_seed)
    total_windows = cfg.gradient_accumulation_steps * cfg.batch_size
    starts = torch.randint(
        0, len(data) - cfg.context_len, (total_windows,), generator=g
    ).tolist()

    def _window(i):
        x = torch.from_numpy(data[i : i + cfg.context_len].astype(np.int64))
        y = torch.from_numpy(data[i + 1 : i + 1 + cfg.context_len].astype(np.int64))
        return x, y

    micro_batches = []
    for b in range(cfg.gradient_accumulation_steps):
        chunk = starts[b * cfg.batch_size : (b + 1) * cfg.batch_size]
        xs = torch.stack([_window(i)[0] for i in chunk])
        ys = torch.stack([_window(i)[1] for i in chunk])
        if "cuda" in device:
            xs = xs.pin_memory().to(device, non_blocking=True)
            ys = ys.pin_memory().to(device, non_blocking=True)
        else:
            xs = xs.to(device)
            ys = ys.to(device)
        micro_batches.append((xs, ys))

    X_all = torch.cat([b[0] for b in micro_batches], dim=0)
    Y_all = torch.cat([b[1] for b in micro_batches], dim=0)
    return micro_batches, (X_all, Y_all), data_dir


def create_train_state(cfg, device):
    gpt_conf = GPTConfig(
        context_len=cfg.context_len,
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        embd_dim=cfg.embd_dim,
        bias=cfg.use_bias,
        init_var=cfg.init_var,
        use_flash=cfg.use_flash,
    )

    model = GPT(gpt_conf)

    optim = model.configure_optimizers(
        learning_rate=cfg.lr_init,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        device_type=cfg.device_type,
    )

    loss_fn = CrossEntropyLoss()
    return model, loss_fn, optim


def train_and_evaluate(cfg, device):
    model, loss_fn, optim = create_train_state(cfg, device)

    num_params, embd_params = model.get_num_params()
    print(f"number of parameters: {num_params / 1e6:.2f}M")
    print(f"number of embd params: {embd_params / 1e6:.2f}M")

    train_results = DataStorage(
        columns=["step", "lr_step", "loss_step"],
        num_params=num_params,
        embd_params=embd_params,
    )
    forward_results = DataStorage(
        columns=["step", "lr_step", "lr_lower", "lr_upper", "num_iters_critical"],
        num_params=num_params,
        embd_params=embd_params,
    )
    sharpness_results = DataStorage(
        columns=[
            "step",
            "lr_step",
            "pre_sharpness",
            "pre_num_iters",
            "sharpness",
            "num_iters",
        ],
        num_params=num_params,
        embd_params=embd_params,
    )
    eval_results = DataStorage(
        columns=["step", "lr_step", "train_loss", "val_loss"],
        num_params=num_params,
        embd_params=embd_params,
    )

    model.to(device)
    for state in optim.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

    if cfg.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    micro_batches, (X_all, Y_all), data_dir = build_full_batch(cfg, device)
    full_batch_size = X_all.shape[0]
    print(
        f"full batch: {cfg.gradient_accumulation_steps} micro-batches of "
        f"{cfg.batch_size}x{cfg.context_len} = {full_batch_size} windows"
    )

    lr_guess = cfg.lr_peak
    lr_step = warmup_stable_decay(
        step=0,
        init_value=cfg.lr_init,
        peak_value=cfg.lr_peak,
        min_value=cfg.lr_min,
        num_steps=cfg.num_steps,
        warmup_steps=cfg.warmup_steps,
        stable_steps=cfg.stable_steps,
        warmup_exponent=cfg.warmup_exponent,
        decay_schedule_name=cfg.decay_schedule_name,
        decay_exponent=cfg.decay_exponent,
    )
    ctx = (
        nullcontext()
        if cfg.device_type == "cpu"
        else torch.amp.autocast(device_type=cfg.device_type, dtype=cfg.ptdtype)
    )

    eigvec = None
    pre_eigvec = None
    crit_thresh = (2 + 2 * cfg.beta1) / (1 - cfg.beta1)

    print(f"Recompute Grads: {cfg.recompute_grads}")

    use_wandb = cfg.use_wandb and WANDB_AVAILABLE
    if cfg.use_wandb and not WANDB_AVAILABLE:
        print("wandb requested but not installed; skipping wandb logging.")
    if use_wandb:
        run_base = get_base_filename(cfg, cfg.num_steps)
        run_id = hashlib.md5(run_base.encode()).hexdigest()
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name or run_base[:128],
            id=run_id,
            resume="allow",
            config={**vars(cfg), "num_params": num_params, "embd_params": embd_params},
        )

    for step in range(0, cfg.num_steps + 1):

        wandb_metrics = {} if use_wandb else None

        if step % cfg.sharpness_interval == 0:
            (lr_lower, lr_upper), num_iters_critical = compute_critical_learning_rate(
                ctx=ctx,
                model=model,
                loss_fn=loss_fn,
                optim=optim,
                batch=(X_all, Y_all),
                lr_guess=lr_guess,
                tol_power=cfg.crit_lr_tol_power,
                recompute_grads=cfg.recompute_grads,
                num_microbatches=cfg.gradient_accumulation_steps,
            )
            # guard against degenerate (underflowed / non-converged) search results
            # — compute_critical_learning_rate requires lr_guess > 0 next call
            lr_guess = lr_lower if lr_lower > 0 else cfg.lr_peak

            forward_results.add_entry(
                step=step,
                lr_step=lr_step,
                lr_lower=lr_lower,
                lr_upper=lr_upper,
                num_iters_critical=num_iters_critical,
            )
            print(
                f"LR Range: {lr_lower:0.1e}, {lr_upper:0.1e} computed in "
                f"{num_iters_critical} steps"
            )
            if use_wandb:
                wandb_metrics["lr_lower"] = lr_lower
                wandb_metrics["lr_upper"] = lr_upper
                wandb_metrics["num_iters_critical"] = num_iters_critical

            optim.zero_grad(set_to_none=True)

            pre_sharpness_step, pre_eigvec, num_iters_pre_sharpness = (
                sharpness_utils.get_pre_sharpness_lobpcg(
                    model,
                    loss_fn,
                    optim,
                    (X_all, Y_all),
                    eigvecs=pre_eigvec,
                    tol=cfg.sharpness_tol,
                    max_iters=cfg.max_iters,
                    num_microbatches=cfg.gradient_accumulation_steps,
                )
            )
            print(
                f"Pre-sharpness: {pre_sharpness_step:0.4f} computed in "
                f"{num_iters_pre_sharpness} steps, critical LR: "
                f"{crit_thresh / pre_sharpness_step:0.1e}"
            )
            sharpness_step, eigvec, num_iters_sharpness = (
                sharpness_utils.get_sharpness_lobpcg(
                    model,
                    loss_fn,
                    (X_all, Y_all),
                    eigvecs=eigvec,
                    tol=cfg.sharpness_tol,
                    max_iters=cfg.max_iters,
                    num_microbatches=cfg.gradient_accumulation_steps,
                )
            )
            print(
                f"Sharpness: {sharpness_step:0.4f} computed in "
                f"{num_iters_sharpness} steps"
            )

            sharpness_results.add_entry(
                step=step,
                lr_step=lr_step,
                pre_sharpness=pre_sharpness_step,
                pre_num_iters=num_iters_pre_sharpness,
                sharpness=sharpness_step,
                num_iters=num_iters_sharpness,
            )
            if use_wandb:
                wandb_metrics["pre_sharpness"] = pre_sharpness_step
                wandb_metrics["pre_num_iters"] = num_iters_pre_sharpness
                wandb_metrics["sharpness"] = sharpness_step
                wandb_metrics["num_iters_sharpness"] = num_iters_sharpness
                wandb_metrics["critical_lr_sharpness"] = (
                    crit_thresh / pre_sharpness_step
                )

            optim.zero_grad(set_to_none=True)

        if step % cfg.eval_interval == 0:
            losses = estimate_loss(
                ctx,
                model,
                loss_fn,
                cfg.eval_steps,
                data_dir,
                cfg.context_len,
                cfg.batch_size,
                device,
            )
            eval_results.add_entry(
                step=step,
                lr_step=lr_step,
                train_loss=losses["train"].item(),
                val_loss=losses["val"].item(),
            )
            if use_wandb:
                wandb_metrics["train_loss"] = losses["train"].item()
                wandb_metrics["val_loss"] = losses["val"].item()

            train_results.save_to_csv(cfg.train_path)
            forward_results.save_to_csv(cfg.forward_path)
            eval_results.save_to_csv(cfg.evals_path)
            sharpness_results.save_to_csv(cfg.sharpness_path)

        optim.zero_grad(set_to_none=True)

        lr_step = warmup_stable_decay(
            step=step,
            init_value=cfg.lr_init,
            peak_value=cfg.lr_peak,
            min_value=cfg.lr_min,
            num_steps=cfg.num_steps,
            warmup_steps=cfg.warmup_steps,
            stable_steps=cfg.stable_steps,
            warmup_exponent=cfg.warmup_exponent,
            decay_schedule_name=cfg.decay_schedule_name,
            decay_exponent=cfg.decay_exponent,
        )
        for param_group in optim.param_groups:
            param_group["lr"] = lr_step * param_group.get("lr_scale", 1.0)

        for X, Y in micro_batches:
            with ctx:
                logits = model(X)
                loss = loss_fn(logits, Y) / cfg.gradient_accumulation_steps
            loss.backward()

        if cfg.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        lossf = loss.item() * cfg.gradient_accumulation_steps
        print(f"step: {step}, lr: {lr_step:0.1e}, loss: {lossf:.4f}")
        train_results.add_entry(step=step, lr_step=lr_step, loss_step=lossf)

        if use_wandb:
            wandb_metrics["lr_step"] = lr_step
            wandb_metrics["loss_step"] = lossf
            wandb.log(wandb_metrics, step=step)

        optim.step()
        # don't free up gradients; reused by next step's critical LR estimation

    train_results.save_to_csv(cfg.train_path)
    forward_results.save_to_csv(cfg.forward_path)
    eval_results.save_to_csv(cfg.evals_path)
    sharpness_results.save_to_csv(cfg.sharpness_path)

    if use_wandb:
        wandb.finish()


### CONFIG ###

parser = argparse.ArgumentParser(description="Full-batch GD debug run on Shakespeare")
parser.add_argument("--dtype", type=str, default="float32")

### dataset
parser.add_argument("--dataset_name", type=str, default="shakespeare")
parser.add_argument("--vocab_size", type=int, default=50304)
parser.add_argument(
    "--max_train_tokens",
    type=int,
    default=200_000,
    help="Truncate train.bin to this many tokens before sampling windows. 0 = use all.",
)
parser.add_argument(
    "--data_seed",
    type=int,
    default=0,
    help="Seed for the one-time window sampling that defines the fixed full batch.",
)

### model
parser.add_argument("--model_name", type=str, default="gpt")
parser.add_argument("--init_var", type=float, default=1.0)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--head_dim", type=int, default=64)
parser.add_argument("--embd_dim", type=int, default=256)
parser.add_argument("--bias", type=str, default="False")
parser.add_argument("--context_len", type=int, default=256)
parser.add_argument("--compile", type=lambda x: x.lower() == "true", default=True)
parser.add_argument("--use_flash", type=lambda x: x.lower() == "true", default=False)

### optimization
parser.add_argument("--optim_name", type=str, default="AdamW")
parser.add_argument("--lr_init", type=float, default=0.0)
parser.add_argument("--lr_peak", type=float, default=1e-05)
parser.add_argument(
    "--lr_min_factor",
    type=lambda x: float("inf") if x.lower() == "inf" else float(x),
    default=float("inf"),
)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.95)
parser.add_argument("--eps", type=float, default=1e-08)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--num_steps", type=int, default=10_000)
parser.add_argument("--warmup_steps", type=int, default=1_000)
parser.add_argument("--warmup_exponent", type=float, default=1.0)
parser.add_argument("--stable_steps", type=int, default=8_000)
parser.add_argument("--decay_schedule_name", type=str, default="polynomial")
parser.add_argument("--decay_exponent", type=float, default=1.0)
parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--grad_clip", type=float, default=0.0)

### evaluation
parser.add_argument("--results_dir", type=str, default="shakespeare_fullgd_results")
parser.add_argument("--log_interval", type=int, default=1)
parser.add_argument("--eval_interval", type=int, default=100)
parser.add_argument("--eval_steps", type=int, default=10)
parser.add_argument("--verbose", type=bool, default=False)

### sharpness estimation
parser.add_argument("--sharpness_interval", type=int, default=10)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--max_iters", type=int, default=200)
parser.add_argument("--sharpness_tol", type=float, default=1e-11)

### critical learning rate estimation
parser.add_argument("--crit_lr_tol_power", type=int, default=4)
parser.add_argument(
    "--recompute_grads", type=lambda x: x.lower() == "true", default=True
)

### wandb logging
parser.add_argument(
    "--use_wandb", type=lambda x: x.lower() == "true", default=False
)
parser.add_argument("--wandb_project", type=str, default="scalable-curvature")
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)


cfg = parser.parse_args()

cfg.use_bias = cfg.bias == "True"
cfg.dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
cfg.embd_dim = cfg.head_dim * cfg.num_heads
cfg.lr_min = cfg.lr_peak / cfg.lr_min_factor

device = "cuda"
torch.manual_seed(1337)

cfg.device_type = "cuda" if "cuda" in device else "cpu"
cfg.ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[cfg.dtype]

os.makedirs(cfg.results_dir, exist_ok=True)

base_filename = get_base_filename(cfg, cfg.num_steps)
cfg.train_path = os.path.join(cfg.results_dir, f"train_{base_filename}.csv")
cfg.evals_path = os.path.join(cfg.results_dir, f"eval_{base_filename}.csv")
cfg.forward_path = os.path.join(cfg.results_dir, f"forward_{base_filename}.csv")
cfg.sharpness_path = os.path.join(cfg.results_dir, f"sharpness_{base_filename}.csv")

train_and_evaluate(cfg, device)
