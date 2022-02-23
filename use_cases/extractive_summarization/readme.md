# PreSumm

**This code is for EMNLP 2019 paper [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)**

**Updates Jan 22 2020**: Now you can **Summarize Raw Text Input!**. Swith to the dev branch, and use `-mode test_text` and use `-text_src $RAW_SRC.TXT` to input your text file. Please still use master branch for normal training and evaluation, dev branch should be only used for test_text mode.
* abstractive use -task abs, extractive use -task ext
* use `-test_from $PT_FILE$` to use your model checkpoint file.
* Format of the source text file:
  * For **abstractive summarization**, each line is a document.
  * If you want to do **extractive summarization**, please insert ` [CLS] [SEP] ` as your sentence boundaries.
* There are example input files in the [raw_data directory](https://github.com/nlpyang/PreSumm/tree/dev/raw_data)
* If you also have reference summaries aligned with your source input, please use `-text_tgt $RAW_TGT.TXT` to keep the order for evaluation.


Results on CNN/DailyMail (20/8/2019):


<table class="tg">
  <tr>
    <th class="tg-0pky">Models</th>
    <th class="tg-0pky">ROUGE-1</th>
    <th class="tg-0pky">ROUGE-2</th>
    <th class="tg-0pky">ROUGE-L</th>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="4">Extractive</td>
  </tr>
  <tr>
    <td class="tg-0pky">TransformerExt</td>
    <td class="tg-0pky">40.90</td>
    <td class="tg-0pky">18.02</td>
    <td class="tg-0pky">37.17</td>
  </tr>
  <tr>
    <td class="tg-0pky">BertSumExt</td>
    <td class="tg-0pky">43.23</td>
    <td class="tg-0pky">20.24</td>
    <td class="tg-0pky">39.63</td>
  </tr>
  <tr>
    <td class="tg-0pky">BertSumExt (large)</td>
    <td class="tg-0pky">43.85</td>
    <td class="tg-0pky">20.34</td>
    <td class="tg-0pky">39.90</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="4">Abstractive</td>
  </tr>
  <tr>
    <td class="tg-0lax">TransformerAbs</td>
    <td class="tg-0lax">40.21</td>
    <td class="tg-0lax">17.76</td>
    <td class="tg-0lax">37.09</td>
  </tr>
  <tr>
    <td class="tg-0lax">BertSumAbs</td>
    <td class="tg-0lax">41.72</td>
    <td class="tg-0lax">19.39</td>
    <td class="tg-0lax">38.76</td>
  </tr>
  <tr>
    <td class="tg-0lax">BertSumExtAbs</td>
    <td class="tg-0lax">42.13</td>
    <td class="tg-0lax">19.60</td>
    <td class="tg-0lax">39.18</td>
  </tr>
</table>

**Python version**: This code is in Python3.6

**Package Requirements**: torch==1.1.0 pytorch_transformers tensorboardX multiprocess pyrouge



**Updates**: For encoding a text longer than 512 tokens, for example 800. Set max_pos to 800 during both preprocessing and training.


Some codes are borrowed from ONMT(https://github.com/OpenNMT/OpenNMT-py)

## Trained Models
[CNN/DM BertExt](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ)

[CNN/DM BertExtAbs](https://drive.google.com/open?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr)

[CNN/DM TransformerAbs](https://drive.google.com/open?id=1yLCqT__ilQ3mf5YUUCw9-UToesX5Roxy)

[XSum BertExtAbs](https://drive.google.com/open?id=1H50fClyTkNprWJNh10HWdGEdDdQIkzsI)

## System Outputs

[CNN/DM and XSum](https://drive.google.com/file/d/1kYA384UEAQkvmZ-yWZAfxw7htCbCwFzC) 

## Data Preparation For XSum
[Pre-processed data](https://drive.google.com/open?id=1BWBN1coTWGBqrWoOfRc5dhojPHhatbYs)


## Data Preparation For CNN/Dailymail
### Option 1: download the processed data

[Pre-processed data](https://drive.google.com/open?id=1DN7ClZCCXsk2KegmC6t4ClBwtAf5galI)

unzip the zipfile and put all `.pt` files into `bert_data`

### Option 2: process the data yourself

#### Step 1 Download Stories
Download and unzip the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put all  `.story` files in one directory (e.g. `../raw_stories`)

####  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-full-2017-06-09` directory. 

####  Step 3. Sentence Splitting and Tokenization

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
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH -lower -n_cpus 1 -log_file ../logs/preprocess.log -max_src_nsents 500000 -min_src_nsents 1 -min_src_ntokens_per_sent 0 -max_src_ntokens_per_sent 512 -min_tgt_ntokens 0 -max_tgt_ntokens 200000 
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

