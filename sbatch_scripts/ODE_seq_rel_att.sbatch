#!/bin/bash
#SBACTH --error=../logs/ODE_seq_rel_att-%j.err   # Файл с ошибками
#SBATCH --output=../logs/ODE_seq_rel_att-%j.log  # Файл для вывода результатов
#SBATCH --time=2-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --max_epoch 200 --exp_name ODE_seq_rel_att --tasks "ode1" --reload_data "ode1,../data/ode1.train,../data/ode1.valid,../data/ode1.test" --dump_path '../dumped/' --env_base_seed 0 --torch_n_random_seed 0 --use_pos_embeddings_E False --use_pos_embeddings_D False --max_relative_pos 32 --use_neg_dist True --use_encdec_seq_rel_att True


