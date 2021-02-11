#!/bin/bash
#SBACTH --error=../logs/debug-%j.err   # Файл с ошибками
#SBATCH --output=../logs/debug-%j.log  # Файл для вывода результатов
#SBATCH --time=01:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --exp_name debug --tasks "prim_fwd" --reload_data "prim_fwd,../data/prim_fwd.train_clean,../data/prim_fwd.valid,../data/prim_fwd.test" --dump_path '../dumped/' --use_tree_rel_att 'mult1' --tree_rel_vocab_size 26316 --reload_rel_matrices "prim_fwd,../data/rel_matrix_train_clean.json,../data/rel_matrix_valid.json,../data/rel_matrix_test.json" --rel_vocab_path '../data/rel_vocab.txt'
