#!/bin/bash
#SBACTH --error=../logs/ODE_eval_tree_rel_att-%j.err   # Файл с ошибками
#SBATCH --output=../logs/ODE_eval_tree_rel_att-%j.log  # Файл для вывода результатов
#SBATCH --time=2-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=20          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --cpu True --exp_name ODE_eval_tree_rel_att_nodups --eval_only True --reload_model "/home/kpgelvan/SymbolicMathematics/dumped/ODE_tree_rel_att/277973/checkpoint.pth" --tasks "ode1" --reload_data "ode1,../data/ode1.train_half,../data/ode1.no_dups_valtest,../data/ode1.no_dups_valtest" --dump_path '../dumped/'  --beam_eval true --beam_size 10 --eval_verbose 1 --use_pos_embeddings_D true --use_pos_embeddings_E false --use_tree_rel_att 'mult2' --tree_rel_vocab_size 10000 --reload_rel_matrices "ode1,../data/ODE_rel_matrix_train_half.json,../data/ODE_rel_matrix_no_dups_valtest.json,../data/ODE_rel_matrix_no_dups_valtest.json" --rel_vocab_path '../data/ODE_rel_vocab.txt'
