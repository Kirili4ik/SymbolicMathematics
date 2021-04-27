#!/bin/bash
#SBACTH --error=../logs/eval_pos_emb-%j.err   # Файл с ошибками
#SBATCH --output=../logs/eval_pos_emb-%j.log  # Файл для вывода результатов
#SBATCH --time=1-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --exp_name eval_pos_emb --eval_only True --reload_model "/home/kpgelvan/SymbolicMathematics/dumped/pos_emb_0202/269255/checkpoint.pth" --tasks "prim_fwd" --reload_data "prim_fwd,../data/prim_fwd.train_clean,../data/prim_fwd.no_dups_valtest,../data/prim_fwd.no_dups_valtest" --dump_path '../dumped/' --beam_eval True --beam_size 10 --eval_verbose 1 --use_pos_embeddings_E True --use_pos_embeddings_D True

