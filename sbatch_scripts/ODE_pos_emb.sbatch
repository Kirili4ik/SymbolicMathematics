#!/bin/bash
#SBACTH --error=../logs/ODE_pos_emb-%j.err   # Файл с ошибками
#SBATCH --output=../logs/ODE_pos_emb-%j.log  # Файл для вывода результатов
#SBATCH --time=2-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --max_epoch 50 --exp_name ODE_pos_emb_small --tasks "ode1" --reload_data "ode1,../data/ode1.train,../data/ode1.valid,../data/ode1.test" --dump_path '../dumped/' --env_base_seed -1 --torch_n_random_seed -1 --use_pos_embeddings_E True --use_pos_embeddings_D True
