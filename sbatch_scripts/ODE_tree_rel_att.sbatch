#!/bin/bash
#SBACTH --error=../logs/ODE_tree_rel_att-%j.err   # Файл с ошибками
#SBATCH --output=../logs/ODE_tree_rel_att-%j.log  # Файл для вывода результатов
#SBATCH --time=3-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --max_epoch 200 --exp_name ODE_tree_rel_att --tasks "ode1" --reload_data "ode1,../data/ode1.train_half,../data/ode1.valid,../data/ode1.test" --dump_path '../dumped/' --use_pos_embeddings_E False --use_pos_embeddings_D True --use_tree_rel_att 'mult2' --tree_rel_vocab_size 10000 --reload_rel_matrices "ode1,../data/ODE_rel_matrix_train_half.json,../data/ODE_rel_matrix_valid.json,../data/ODE_rel_matrix_test.json" --rel_vocab_path '../data/ODE_rel_vocab.txt'
