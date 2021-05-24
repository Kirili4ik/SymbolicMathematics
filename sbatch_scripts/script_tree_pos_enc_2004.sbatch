#!/bin/bash
#SBACTH --error=../logs/tree_pos_enc_2004-%j.err   # Файл с ошибками
#SBATCH --output=../logs/tree_pos_enc_2004-%j.log  # Файл для вывода результатов
#SBATCH --time=5-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --exp_name tree_pos_enc_small --max_epoch 100 --tasks "prim_fwd" --reload_data "prim_fwd,../data/prim_fwd.train_clean,../data/prim_fwd.valid,../data/prim_fwd.test" --dump_path '../dumped/' --env_base_seed -1 --torch_n_random_seed -1 --use_pos_embeddings_E False --use_pos_embeddings_D True --use_tree_pos_enc_E True --use_tree_pos_enc_D False --max_path_width 4 --max_path_depth 16 --reload_root_paths "prim_fwd,../data/root_paths_train_clean_QA.json,../data/root_paths_valid_QA.json,../data/root_paths_test_QA.json"
