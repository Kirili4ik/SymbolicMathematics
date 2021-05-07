#!/bin/bash
#SBACTH --error=../logs/ODE_tree_pos_enc-%j.err   # Файл с ошибками
#SBATCH --output=../logs/ODE_tree_pos_enc-%j.log  # Файл для вывода результатов
#SBATCH --time=2-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --max_epoch 200 --exp_name ODE_tree_pos_enc --tasks "ode1" --reload_data "ode1,../data/ode1.train,../data/ode1.valid,../data/ode1.test" --dump_path '../dumped/' --env_base_seed 0 --torch_n_random_seed 0 --use_pos_embeddings_E False --use_pos_embeddings_D False --use_tree_pos_enc_E True --use_tree_pos_enc_D True --max_path_width 4 --max_path_depth 16 --reload_root_paths "ode1,../data/ODE_root_paths_QAtrain.json,../data/ODE_root_paths_QAvalid.json,../data/ODE_root_paths_QAtest.json"

