#!/bin/bash
#SBACTH --error=logs/first_base_eval-%j.err   # Файл с ошибками
#SBATCH --output=logs/first_base_eval-%j.log  # Файл для вывода результатов
#SBATCH --time=23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --exp_name first_baseline_eval --eval_only true --reload_model '' --tasks "prim_fwd" --reload_data "prim_fwd,data/prim_fwd.train,data/prim_fwd.valid,data/prim_fwd.test" --beam_eval true --beam_size 10 --beam_length_penalty 1.0 --beam_early_stopping 1 --eval_verbose 1 --eval_verbose_print false
