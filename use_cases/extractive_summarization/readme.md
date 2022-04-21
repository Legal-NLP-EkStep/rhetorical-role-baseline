# Extractive Summarization of Legal docs using rhetorical roles

## 1. What is Extractive Text Summarization?

Extractive Text Summarization is the task of extracting important information or sentence from the given text.
Extractive summarizers are so-called because they select sentences from the originally given text passage to create the
summary.

## 2. Extractive summarization using Rhetorical Roles

We wanted to see how Rhetorical Roles could help generate better summaries. Towards this goal, we passed the sentences
through bertsumm where sentences were selected to create the summaries. To see the impact of rhetorical roles we
provided sentences with the predicted rhetorical roles from our model. This helped in generating summaries of various
roles separately. This approach improved the generated summaries.

# Setup

**Python version**: This code is in Python3.7

```pip install -r requirements.txt```

## Data Preparation

### Option 1: download the processed data

[Pre-processed data](https://)

unzip the zipfile

### Option 2: process the data yourself

#### Step 1 Download Stories

Download and unzip the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put
all  `.story` files in one directory (e.g. `../raw_stories`)

#### Note: Replace with your data

#### Step 2. Download Stanford CoreNLP

We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip
it. Then add the following command to your bash_profile:

```
export CLASSPATH=/path/to/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar
```

replacing `/path/to/` with the path to where you saved the `stanford-corenlp-full-2017-06-09` directory.

#### Step 3. Sentence Splitting and Tokenization

```
python preprocess.py -mode tokenize -raw_path RAW_PATH -save_path TOKENIZED_PATH
```

* `RAW_PATH` is the directory containing story files (`../raw_stories`), `JSON_PATH` is the target directory to save the generated json files (`../merged_stories_tokenized`)


####  Step 4. Format to Simpler Json Files
 
```
python preprocess.py -mode format_to_lines -raw_path RAW_PATH -save_path JSON_PATH -n_cpus 1 -use_bert_basic_tokenizer false -map_path MAP_PATH
```

* `RAW_PATH` is the directory containing tokenized files (`../merged_stories_tokenized`), `JSON_PATH` is the target directory to save the generated json files (`../json_data/cnndm`), `MAP_PATH` is the  directory containing the urls files (`../urls`)

####  Step 5. Format to PyTorch Files
```
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH -lower -n_cpus 1 -log_file ../logs/preprocess.log max_src_nsents 500000 -min_src_nsents 1 -min_src_ntokens_per_sent 0 -max_src_ntokens_per_sent 512 -min_tgt_ntokens 0 -max_tgt_ntokens 200000 
```

* `JSON_PATH` is the directory containing json files (`../json_data`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`)

## Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

### Extractive Setting

```
python3 train.py -task ext -mode train -bert_data_path BERT_DATA_PATH -ext_dropout 0.1 -model_path MODEL_PATH -lr 2e-3 -report_every 50 -save_checkpoint_steps 500 -batch_size 5000 -train_steps 5000 -accum_count 2 -log_file ../logs/ext_bert_3jan -use_interval true -warmup_steps 1000 -max_pos 512 -use_rhetorical_roles true
```

## Model Evaluation
```
 python3 train.py -task ext -mode test -test_from /data/bertsum/model.pt -batch_size 5000 -test_batch_size 1 -bert_data_path BERT_DATA_PATH -log_file ../logs/bertsum -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 2000 -alpha 0.95 -min_length 0 -result_path ../logs/ -use_rhetorical_roles true -rogue_exclude_roles_not_in_test true -add_additional_mandatory_roles_to_summary true  -use_adaptive_summary_sent_percent true 
```
* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries (this will take a while)

