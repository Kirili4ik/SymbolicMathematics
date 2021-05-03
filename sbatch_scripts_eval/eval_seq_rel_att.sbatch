#!/bin/bash
#SBACTH --error=../logs/eval_seq_rel_att-%j.err   # Файл с ошибками
#SBATCH --output=../logs/eval_seq_rel_att-%j.log  # Файл для вывода результатов
#SBATCH --time=1-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

source activate sym-math
srun python ../main.py --exp_name eval_seq_rel_att --eval_only True --reload_model "/home/kpgelvan/SymbolicMathematics/dumped/seq_rel_att_0202/272617/checkpoint.pth" --tasks "prim_fwd" --reload_data "prim_fwd,../data/prim_fwd.train_clean,../data/prim_fwd.no_dups_valtest,../data/prim_fwd.no_dups_valtest" --dump_path '../dumped/' --beam_eval true --beam_size 10 --eval_verbose 1 --use_pos_embeddings_E false --use_pos_embeddings_D false --max_relative_pos 250 --use_neg_dist True --use_encdec_seq_rel_att True
