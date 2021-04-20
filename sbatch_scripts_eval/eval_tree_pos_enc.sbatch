#!/bin/bash
#SBACTH --error=../logs/eval_tree_pos_enc-%j.err   # Файл с ошибками
#SBATCH --output=../logs/eval_tree_pos_enc-%j.log  # Файл для вывода результатов
#SBATCH --time=4-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --exp_name eval_tree_pos_enc --eval_only True --reload_model "/home/kpgelvan/SymbolicMathematics/dumped/tree_pos_enc_1204/264200/checkpoint.pth" --tasks "prim_fwd" --reload_data "prim_fwd,../data/prim_fwd.train_clean,../data/prim_fwd.no_dups_valtest,../data/prim_fwd.no_dups_valtest" --dump_path '../dumped/' --beam_eval true --beam_size 10 --eval_verbose 1 --use_pos_embeddings_E False --use_pos_embeddings_D True --use_tree_pos_enc True --max_path_width 4 --max_path_depth 16 --reload_root_paths "prim_fwd,../data/root_paths_train_clean.json,../data/root_paths_no_dups_valtest.json,../data/root_paths_no_dups_valtest.json"
