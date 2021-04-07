#!/bin/bash
#SBACTH --error=../logs/eval_tree_rel_att-%j.err   # Файл с ошибками
#SBATCH --output=../logs/eval_tree_rel_att-%j.log  # Файл для вывода результатов
#SBATCH --time=2-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --exp_name eval_tree_rel_att --eval_only True --reload_model "/home/kpgelvan/SymbolicMathematics/dumped/tree_rel_att_0303/232595/checkpoint.pth" --tasks "prim_fwd" --reload_data "prim_fwd,../data/prim_fwd.train,../data/prim_fwd.valid,../data/prim_fwd.test" --dump_path '../dumped/' --env_base_seed 0 --torch_n_random_seed 0 --beam_eval true --beam_size 10 --eval_verbose 1 --use_pos_embeddings_D true --use_pos_embeddings_E false --use_tree_rel_att 'mult2' --tree_rel_vocab_size 2000 --reload_rel_matrices "prim_fwd,../data/rel_matrix_train_clean.json,../data/rel_matrix_valid.json,../data/rel_matrix_test.json" --rel_vocab_path '../data/rel_vocab.txt'
