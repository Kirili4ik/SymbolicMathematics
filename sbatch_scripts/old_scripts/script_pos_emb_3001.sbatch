#!/bin/bash
#SBACTH --error=../logs/pos_emb_3001-%j.err   # Файл с ошибками
#SBATCH --output=../logs/pos_emb_3001-%j.log  # Файл для вывода результатов
#SBATCH --time=2-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --exp_name pos_emb_3001 --tasks "prim_fwd" --reload_data "prim_fwd,../data/prim_fwd.train,../data/prim_fwd.valid,../data/prim_fwd.test" --dump_path '../dumped/' --use_pos_embeddings True --max_relative_pos 0 --use_neg_dist False

