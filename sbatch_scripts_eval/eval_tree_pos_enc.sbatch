#!/bin/bash
#SBACTH --error=../logs/eval_tree_pos_enc_2004_nodups-%j.err   # Файл с ошибками
#SBATCH --output=../logs/eval_tree_pos_enc_2004_nodups-%j.log  # Файл для вывода результатов
#SBATCH --time=15:00:00            # Максимальное время выполнения
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=15          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --cpu True --exp_name eval_tree_pos_enc_2004_nodups --eval_only True --reload_model "/home/kpgelvan/SymbolicMathematics/dumped/tree_pos_enc_2004/268765/checkpoint.pth" --tasks "prim_fwd" --reload_data "prim_fwd,../data/prim_fwd.train_clean,../data/prim_fwd.no_dups_valtest,../data/prim_fwd.no_dups_valtest" --dump_path '../dumped/' --beam_eval true --beam_size 10 --eval_verbose 1 --use_pos_embeddings_E False --use_pos_embeddings_D False --use_tree_pos_enc_E True --use_tree_pos_enc_D True --max_path_width 4 --max_path_depth 16 --reload_root_paths "prim_fwd,../data/root_paths_train_clean_QA.json,../data/root_paths_no_dups_valtest_QA.json,../data/root_paths_no_dups_valtest_QA.json"
