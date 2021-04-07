#!/bin/sh
#SBACTH --error=../logs/prim_FWD_no_dups1-%j.err   # Файл с ошибками
#SBATCH --output=../logs/prim_FWD_no_dups1-%j.log  # Файл для вывода результатов
#SBATCH --time=2-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=10          # Требуемое кол-во CPU


source activate sym-math
srun python ../main.py --export_data True --batch_size 32 --cpu True --exp_name prim_FWD_no_dups --num_workers 100 --tasks prim_fwd --env_base_seed -1 --n_variables 1 --n_coefficients 0 --leaf_probs "0.75,0,0.25,0" --max_ops 15 --max_int 5 --positive False --max_len 512 --operators "add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1"
