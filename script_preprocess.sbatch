#!/bin/bash
#SBACTH --error=logs/preprocess-%j.err   # Файл с ошибками
#SBATCH --output=logs/preprocess-%j.log  # Файл для вывода результатов
#SBATCH --time=2-23:00:00            # Максимальное время выполнения
#SBATCH --gpus=0                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=1          # Требуемое кол-во CPU

source activate sym-math
srun python preprocess_script.py
