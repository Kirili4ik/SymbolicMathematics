#!/bin/bash
#SBACTH --error=logs/fp16-%j.err   # Файл с ошибками
#SBATCH --output=logs/fp16-%j.log  # Файл для вывода результатов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python main.py --exp_name first_train --fp16 true --amp 2 --tasks "prim_fwd" --reload_data "prim_fwd,data/prim_fwd.train,data/prim_fwd.valid,data/prim_fwd.test" --emb_dim 1024 --n_enc_layers 6 --n_dec_layers 6 --n_heads 8


