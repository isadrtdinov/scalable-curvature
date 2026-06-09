#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

#SBATCH --job-name=eval-curvature
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out

# Environment setup
source venv/bin/activate

# Single-process job (no DDP): eval_curvature_ckpts.py iterates ckpts sequentially
# on one GPU. --num_microbatches splits the HVP / preconditioner-grad / critical-LR
# forward+backward into N passes so a large --batch_size fits in memory.
python language/eval_curvature_ckpts.py \
    --ckpt_dir ./checkpoints \
    --dataset_name openwebtext \
    --batch_size 128 \
    --num_microbatches 8 \
    --results_dir curvature_results \
    --output_csv curvature.csv \
    --max_iters 500 \
    --sharpness_tol 1e-09 \
    --crit_lr_tol_power 4 \
    --lr_guess 1e-04
