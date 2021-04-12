# A Mathematical Exploration of Why Language Models Help Solve Downstream Tasks
This repository contains the code for [this paper](https://arxiv.org/abs/2010.03648). Our repository is a fork of the [HuggingFace transformers repository](https://github.com/huggingface/transformers). 

## Linear Classification Experiments
### Data
Create a directory with the separate folders containing the data for each of the tasks (e.g., `data/sst2`, `data/agnews`). To obtain the data used for the tasks, use the [`text_embedding` repository](https://github.com/NLPrinceton/text_embedding) and **TODO: where did agnews, dbpedia, etc come from**. To use the data retrieval functions in the `text_embedding` repository, run `from documents import <task_name>` and `<task_name>()`. For example, for SST, you can run `from documents import sst` and `(train_docs, train_labels), (test_docs, test_labels) = sst()`. To standardize the format across all datasets, you must save the data in a `.tsv` file with each row having `sentence \t label` (no header row). You must name your folders in a specific way so that the script can match them to the prompt and choose the correct linear classification hyperparameters.
* `sst2`: binary sentence-level SST-2 dataset (6,920 train examples, 1,821 test examples). Use `task_name = sst` to download data from `text_embedding`.
* `sst2_fine`: fine-grained (5-class) SST-2 dataset. Use `task_name = sst_fine` to download data from `text_embedding`.
* `agnews`: AG News dataset (4-class).

**add info about MPQA and CR split**

### Script
The script used to run linear classification on the fixed features produced by pre-trained GPT-2 and BERT models in Table 1 (except the FT column) and Table 2 is `test_downstream.py`. To use it, Then, run `python test_downstream.py --task [TASK_NAME] --data_dir [DATA_DIRECTORY]`. If you want to add the prompt corresponding to the task (i.e., reproduce asterisked results in the tables), then add `--prompt`. Note that the script will generate and save the model features for each of these datasets, and for larger datasets (e.g., AG News), this can result in large (~ 25GB) files. These files will be stored in the directory you pass as the data directory. We also include the option to run classification on the entirety of `p_{f(s)}` by passing `--run_p`, but note that this requires prohibitively expensive amounts of compute for large datasets. 
