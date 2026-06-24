#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=90:00:00

#SBATCH --job-name=fcn-cifar-adamw
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out

# Environment setup
source venv/bin/activate

# Full-batch AdamW debug run on CIFAR-10 with a 4-layer FCN.
# Single GPU; no DDP. The full batch is batch_size=50000 (the entire
# CIFAR-10 training set), so each step is one full-batch gradient.
python image/train_fcn_image_adamw_dir_sharp.py \
    --dataset_name cifar-10 \
    --model_name fcn \
    --width 512 \
    --depth 4 \
    --act_name gelu \
    --loss_name mse \
    --batch_size 50000 \
    --lr_peak 0.03 \
    --lr_min_factor inf \
    --warmup_steps 100 \
    --stable_steps 0 \
    --decay_schedule_name cosine \
    --decay_exponent 1.0 \
    --beta1 0.9 \
    --beta2 0.99 \
    --eps 1e-08 \
    --weight_decay 0.0 \
    --num_steps 10000 \
    --eval_interval 100 \
    --crit_lr_tol_power 4 \
    --recompute_grads true \
    --max_iters 200 \
    --sharpness_tol 1e-11 \
    --subset_size 10000 \
    --results_dir adam_dir_sharp_results \
    --use_wandb true \
    --wandb_project scalable-curvature \
    --wandb_run_name fcn-cifar-adamw
