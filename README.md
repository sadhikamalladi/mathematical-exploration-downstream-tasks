# A Mathematical Exploration of Why Language Models Help Solve Downstream Tasks
This repository contains the code for [this paper](https://arxiv.org/abs/2010.03648). 

## Setting up the Python Environment
We recommend you use conda. Using conda, you can clone the Python environment from the file in the repository by running `conda env create --file environment.yml`.  This will create an environment called `mathematical-downstream` and running `conda activate mathematical-downstream` will activate it. Now you have all of the packages needed to run the code.

## Linear Classification Experiments
The data is available [here](https://drive.google.com/file/d/1LwXpL_YclbHF8rbH7JseX3dGeItauRdM/view?usp=sharing). Create a folder (e.g., `data/`) and download the data from this Google Drive folder, preserving the subfolder structure. For details on preprocessing, please see the README in the folder and Appendix F in the paper.

The script used to run linear classification on the fixed features produced by pre-trained GPT-2 and BERT models in Table 1 (except the FT column) and Table 2 is `test_downstream.py`. To use it, Then, run `python test_downstream.py --task [TASK_NAME] --data_dir [DATA_DIRECTORY]`. The task name corresponds to the folder name in the data (e.g., `sst2`). If you want to add the prompt corresponding to the task (i.e., reproduce asterisked results in the tables), then add `--prompt`. Note that the script will generate and save the model features for each of these datasets, and for larger datasets (e.g., AG News), this can result in large (~ 25GB) files. These files will be stored in the directory you pass as the data directory. We also include the option to run classification on the entirety of `p_{f(s)}` by passing `--run_p`, but note that this requires prohibitively expensive amounts of compute for large datasets. Similarly, we provide the option to save `p_{f(s)}` using `--save_p`.
