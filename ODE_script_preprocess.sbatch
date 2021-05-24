#!/bin/bash
#SBACTH --error=logs/ODE_preprocess-relmat-%j.err   # Файл с ошибками
#SBATCH --output=logs/ODE_preprocess-relmat-%j.log  # Файл для вывода результатов
#SBATCH --time=3-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=3          # Требуемое кол-во CPU

source activate sym-math
srun python ODE_preprocess_build_relmat.py
srun python ODE_preprocess_build_relmat_dict.py
