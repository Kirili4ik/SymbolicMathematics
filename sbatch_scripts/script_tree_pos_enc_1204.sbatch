#!/bin/bash
#SBACTH --error=../logs/tree_pos_enc_0303-%j.err   # Файл с ошибками
#SBATCH --output=../logs/tree_pos_enc_0303-%j.log  # Файл для вывода результатов
#SBATCH --time=4-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --exp_name tree_pos_enc_1204  --tasks "prim_fwd" --reload_data "prim_fwd,../data/prim_fwd.train_clean,../data/prim_fwd.valid,../data/prim_fwd.test" --dump_path '../dumped/' --use_pos_embeddings_E False --use_pos_embeddings_D True --use_tree_pos_enc True --max_path_width 4 --max_path_depth 16  --reload_root_paths "prim_fwd,../data/root_paths_train_clean.json,../data/root_paths_valid.json,../data/root_paths_test.json"
