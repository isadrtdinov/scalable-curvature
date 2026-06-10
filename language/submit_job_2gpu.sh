#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=90:00:00

#SBATCH --job-name=gpt-2gpu
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out

# Environment setup
source venv/bin/activate

# DDP launch: torchrun spawns one process per GPU. The training script reads
# RANK / LOCAL_RANK / WORLD_SIZE from torchrun's env. gradient_accumulation_steps
# is divided by world_size internally, so keep it divisible by 2.
torchrun --standalone --nproc_per_node=2 language/train_gpt_adam_forward_ckpts.py \
    --dataset_name fineweb \
    --num_layers 12 \
    --num_heads 12 \
    --init_var 1.0 \
    --batch_size 16 \
    --gradient_accumulation_steps 64 \
    --lr_peak 1e-05 \
    --lr_min_factor inf \
    --grad_clip 0.0 \
    --warmup_steps 1000 \
    --stable_steps 8000 \
    --decay_schedule_name polynomial \
    --decay_exponent 1.0 \
    --num_steps 10000 \
    --eval_interval 100 \
    --sharpness_interval 10 \
    --ckpt_interval 500 \
    --use_wandb true \
    --wandb_project scalable-curvature \
    --wandb_run_name gpt2-fineweb
