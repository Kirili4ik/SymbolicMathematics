#!/bin/bash
#SBACTH --error=../logs/ODE_eval_tree_pos_enc_nodups-%j.err   # Файл с ошибками
#SBATCH --output=../logs/ODE_eval_tree_pos_enc_nodups-%j.log  # Файл для вывода результатов
#SBATCH --time=15:00:00            # Максимальное время выполнения
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=30          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --cpu True --exp_name ODE_eval_tree_pos_enc_nodups --eval_only True --reload_model "/home/kpgelvan/SymbolicMathematics/dumped/ODE_tree_pos_enc/277972/checkpoint.pth" --tasks "ode1" --reload_data "ode1,../data/ode1.train,../data/ode1.no_dups_valtest,../data/ode1.no_dups_valtest" --dump_path '../dumped/' --beam_eval true --beam_size 10 --eval_verbose 1 --use_pos_embeddings_E False --use_pos_embeddings_D False --use_tree_pos_enc_E True --use_tree_pos_enc_D True --max_path_width 4 --max_path_depth 16 --reload_root_paths "ode1,../data/ODE_root_paths_QAtrain.json,../data/ODE_root_paths_QAno_dups_valtest.json,../data/ODE_root_paths_QAno_dups_valtest.json"
