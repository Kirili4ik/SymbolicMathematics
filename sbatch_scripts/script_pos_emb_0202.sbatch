#!/bin/bash
#SBACTH --error=../logs/pos_emb_0202-%j.err   # Файл с ошибками
#SBATCH --output=../logs/pos_emb_0202-%j.log  # Файл для вывода результатов
#SBATCH --time=4-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --max_epoch 100 --exp_name pos_emb_small --tasks "prim_fwd" --reload_data "prim_fwd,../data/prim_fwd.train_clean,../data/prim_fwd.valid,../data/prim_fwd.test" --dump_path '../dumped/' --env_base_seed -1 --torch_n_random_seed -1 --use_pos_embeddings_E True --use_pos_embeddings_D True
