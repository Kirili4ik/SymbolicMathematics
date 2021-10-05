# Empirical Study of Transformers for Symbolic Mathematics

This is a repository containing code for my Bachelor Thesis made in 2021. The code is based on [this repo](https://github.com/facebookresearch/SymbolicMathematics).

### Abstract:

We investigate whether feeding data structure to the Transformer improves its performance on integration and solving ordinary differential equations (ODEs). We study recently developed tree-based model modifications and compare them. In our experience, the use of these alterations provides no benefit over the base approach. We assume this is due to an uncommonly large amount of data.

📝 [Thesis](https://www.overleaf.com/read/rvncyyqjbbwz), 
👨‍🏫 [Presentation (gdocs)](https://docs.google.com/presentation/d/1CPpGKa_fV8VHdYyUlyoLdI4NnAnVilKW7imYqdCS-oc/edit?usp=sharing)

### Some keypoints:

#### Passing structure to Transformers
![alt text](https://downloader.disk.yandex.ru/preview/c4f3c11d07f494dda0fb5c301b136579554c07019a0d5050d3b4ff2da3917e6d/615d0406/Nvz-RmBdM1EQDrFh0jW-ey4TV4E9jF0xMXrTr5csyFPrE60eyFcXDK6sxx42_QWIAhB56NVLKmpdaqr1_7as9g%3D%3D?uid=0&filename=4samples.jpeg&disposition=inline&hash=&limit=0&content_type=image%2Fjpeg&owner_uid=0&tknv=v2&size=2048x2048)

#### Passing structure to Transformers
![alt text](https://downloader.disk.yandex.ru/preview/0e7c045b97dc9276311d77b02f748632faee82cd8aa12b0d6712c1ef844e1447/615d0450/gESCaBP7OhBPSUBjZ4i7v7oRZ3BhzwM15uhxsl_P9RiiT28y8Wpmx8GCZpYmjJ5IWtqAI90r03FlqwoD9LrUaw%3D%3D?uid=0&filename=image_2021-10-06_00-21-37.png&disposition=inline&hash=&limit=0&content_type=image%2Fpng&owner_uid=0&tknv=v2&size=2048x2048)

##### Problem statement and goal setting
![alt text](https://downloader.disk.yandex.ru/preview/4681ce37ad1e059ba364791ccb01b30332c62d807f02b3fc20f5c3efc8446cce/615d045b/E8ElOzC3TrAlxJXZ5zkouNUD56jZk__yQLCWFou3h4dQff-dtEP06DRn4K_DK-HNdQuNmK2UBMTbeYxYYyKXLw%3D%3D?uid=0&filename=image_2021-10-06_00-23-05.png&disposition=inline&hash=&limit=0&content_type=image%2Fpng&owner_uid=0&tknv=v2&size=2048x2048)

##### Preliminary experiments
![alt text](https://downloader.disk.yandex.ru/preview/f5548144c8a866d791f9d315865f40290a7aadbd7ee6cb99b02b520877e7cb62/615d048c/VrJxqFOQGGLxaRV35vwOcroRZ3BhzwM15uhxsl_P9RhJT9UX0EZmvCGz9oILD--qnbb1QPciz1Attg8n_zUSvA%3D%3D?uid=0&filename=image_2021-10-06_00-24-03.png&disposition=inline&hash=&limit=0&content_type=image%2Fpng&owner_uid=0&tknv=v2&size=2048x2048)

#### Prediction analysis
![alt text](https://downloader.disk.yandex.ru/preview/687d19b6a43f4f5aedb4957f5f449205898155558281296f0dd7a5800f616b46/615d0499/73DxFhWVFOoFlpXrW4mF-roRZ3BhzwM15uhxsl_P9RgZkkptbgo4AFSS0m4-1oe0gboDfzDtplg2WZ-wQUN16Q%3D%3D?uid=0&filename=image_2021-10-06_00-24-20.png&disposition=inline&hash=&limit=0&content_type=image%2Fpng&owner_uid=0&tknv=v2&size=2048x2048)


### How to run

Raw data for training and validation can be found [here](https://github.com/facebookresearch/SymbolicMathematics#datasets-and-trained-models) or [generated](https://github.com/facebookresearch/SymbolicMathematics#data-generation). 
Data preprocessing is done in notebooks/preprocess_notebook.ipynb and notebooks/ODE_preprocess_notebook.ipynb. There can be found functions for:
  1) Deleting found repeating samples
  2) Creating adjacency matrices
  3) Generating paths from root to node

(Also notebooks/ODE_preprocess_notebook-ADJ_MAT.ipynb is for generating adjacency matrices for ODEs separately)

Notebooks by reg \*my_metrics\*.ipynb is for plotting metrics.

The project was done using a server with [Slurm](https://slurm.schedmd.com/documentation.html). Scripts for training and evaluation can be found in sbatch_scripts/ and sbatch_scripts_eval/ folders. Arguments descriptions can be found in [main.py](https://github.com/Kirili4ik/SymbolicMathematics/blob/master/main.py) and in [this repo](https://github.com/facebookresearch/SymbolicMathematics).


Any additional information on running can be found [here](https://github.com/facebookresearch/SymbolicMathematics)
