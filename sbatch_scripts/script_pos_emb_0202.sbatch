#!/bin/bash
#SBACTH --error=../logs/pos_emb_0202-%j.err   # Файл с ошибками
#SBATCH --output=../logs/pos_emb_0202-%j.log  # Файл для вывода результатов
#SBATCH --time=2-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --exp_name pos_emb_0202 --tasks "prim_fwd" --max_epoch 38 --reload_model "/home/kpgelvan/SymbolicMathematics/dumped/pos_emb_0202/194320/checkpoint.pth" --reload_data "prim_fwd,../data/prim_fwd.train,../data/prim_fwd.valid,../data/prim_fwd.test" --dump_path '../dumped/'  --env_base_seed 0 --torch_n_random_seed 0 --use_pos_embeddings_E True --use_pos_embeddings_D True --max_relative_pos 0 --use_neg_dist False
