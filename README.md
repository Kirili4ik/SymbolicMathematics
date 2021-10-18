# Empirical Study of Transformers for Symbolic Mathematics

This is a repository containing code for my Bachelor Thesis made in 2021. The code is based on [this repo](https://github.com/facebookresearch/SymbolicMathematics).

### Abstract:

We investigate whether feeding data structure to the Transformer improves its performance on integration and solving ordinary differential equations (ODEs). We study recently developed tree-based model modifications and compare them. In our experience, the use of these alterations provides no benefit over the base approach. We assume this is due to an uncommonly large amount of data.

📝 [Thesis](https://www.overleaf.com/read/rvncyyqjbbwz), 
👨‍🏫 [Presentation (gdocs)](https://docs.google.com/presentation/d/1CPpGKa_fV8VHdYyUlyoLdI4NnAnVilKW7imYqdCS-oc/edit?usp=sharing)

### Some keypoints:

#### Passing structure to Transformers
![alt text](https://github.com/Kirili4ik/SymbolicMathematics/blob/master/pictures/passing_structure.png)

##### Problem statement and goal setting
![alt text](https://github.com/Kirili4ik/SymbolicMathematics/blob/master/pictures/problem_goal.png)

##### Preliminary experiments
![alt text](https://github.com/Kirili4ik/SymbolicMathematics/blob/master/pictures/preliminary.png)

#### Prediction analysis
![alt text](https://github.com/Kirili4ik/SymbolicMathematics/blob/master/pictures/analysis.jpeg)


### How to run

Raw data for training and validation can be found [here](https://github.com/facebookresearch/SymbolicMathematics#datasets-and-trained-models) or [generated](https://github.com/facebookresearch/SymbolicMathematics#data-generation). 
Data preprocessing is done in `notebooks/preprocess_notebook.ipynb` and `notebooks/ODE_preprocess_notebook.ipynb`, including:
  1) Deleting found repeating samples
  2) Creating adjacency matrices (to a file)
  3) Generating paths from root to node (to a file)

(Also `notebooks/ODE_preprocess_notebook-ADJ_MAT.ipynb` is for generating adjacency matrices for ODEs separately)

Notebooks by reg `*my_metrics*.ipynb` are for plotting metrics.

The project was done using a server with [Slurm](https://slurm.schedmd.com/documentation.html). Scripts for training and evaluation can be found in `sbatch_scripts/` and `sbatch_scripts_eval/` folders respectively. Arguments descriptions can be found in [main.py](https://github.com/Kirili4ik/SymbolicMathematics/blob/master/main.py) or in [this repo](https://github.com/facebookresearch/SymbolicMathematics).


Any additional information on running can be found [here](https://github.com/facebookresearch/SymbolicMathematics)
