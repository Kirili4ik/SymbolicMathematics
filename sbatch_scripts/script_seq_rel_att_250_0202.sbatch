#!/bin/bash
#SBACTH --error=../logs/seq_rel_att_250_0202-%j.err   # Файл с ошибками
#SBATCH --output=../logs/seq_rel_att_250_0202-%j.log  # Файл для вывода результатов
#SBATCH --time=4-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --exp_name seq_rel_att_0202 --tasks "prim_fwd" --max_epoch 17 --reload_model "/home/kpgelvan/SymbolicMathematics/dumped/seq_rel_att_0202/195851/checkpoint.pth" --reload_data "prim_fwd,../data/prim_fwd.train_clean,../data/prim_fwd.valid,../data/prim_fwd.test" --dump_path '../dumped/' --env_base_seed -1 --torch_n_random_seed -1 --use_pos_embeddings_E False --use_pos_embeddings_D False --max_relative_pos 250 --use_neg_dist True
