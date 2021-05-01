#!/bin/bash
#SBACTH --error=../logs/tree_pos_enc_2-%j.err   # Файл с ошибками
#SBATCH --output=../logs/tree_pos_enc_2-%j.log  # Файл для вывода результатов
#SBATCH --time=4-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --exp_name tree_pos_enc_2 --tasks "prim_fwd" --reload_data "prim_fwd,../data/prim_fwd.train_clean,../data/prim_fwd.valid,../data/prim_fwd.test" --dump_path '../dumped/' --env_base_seed 0 --torch_n_random_seed 0 --use_pos_embeddings_E False --use_pos_embeddings_D True --use_tree_pos_enc True --max_path_width 2 --max_path_depth 8 --reload_root_paths "prim_fwd,../data/root_paths_2_train_clean.json,../data/root_paths_2_valid.json,../data/root_paths_2_test.json"
