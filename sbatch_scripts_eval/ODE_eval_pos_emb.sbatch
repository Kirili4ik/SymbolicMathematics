#!/bin/bash
#SBACTH --error=../logs/ODE_eval_pos_emb-%j.err   # Файл с ошибками
#SBATCH --output=../logs/ODE_eval_pos_emb-%j.log  # Файл для вывода результатов
#SBATCH --time=2-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=5          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --cpu True --exp_name ODE_eval_pos_emb_small --eval_only True --reload_model "/home/kpgelvan/SymbolicMathematics/dumped/ODE_pos_emb_small/296490/checkpoint.pth" --tasks "ode1" --reload_data "ode1,../data/ode1.train,../data/ode1.no_dups_valtest,../data/ode1.no_dups_valtest" --dump_path '../dumped/' --beam_eval True --beam_size 10 --eval_verbose 1 --use_pos_embeddings_E True --use_pos_embeddings_D True

