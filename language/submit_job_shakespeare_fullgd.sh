#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=90:00:00

#SBATCH --job-name=gpt-shakes-fullgd
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out

# Environment setup
source venv/bin/activate

# Full-batch GD debug run on Tiny Shakespeare. Single GPU; no DDP. The
# fixed full batch is gradient_accumulation_steps * batch_size windows
# of length context_len, sampled once at init using --data_seed.
python language/train_gpt_full_gd_shakespeare.py \
    --dataset_name shakespeare \
    --num_layers 4 \
    --num_heads 4 \
    --head_dim 64 \
    --context_len 256 \
    --init_var 1.0 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --max_train_tokens 200000 \
    --data_seed 0 \
    --lr_peak 1e-05 \
    --lr_min_factor inf \
    --grad_clip 0.0 \
    --weight_decay 0.0 \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup_steps 1000 \
    --stable_steps 8000 \
    --decay_schedule_name polynomial \
    --decay_exponent 1.0 \
    --num_steps 10000 \
    --eval_interval 100 \
    --sharpness_interval 10 \
    --use_wandb true \
    --wandb_project scalable-curvature \
    --wandb_run_name gpt-shakespeare-fullgd
