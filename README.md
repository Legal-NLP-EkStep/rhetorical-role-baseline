#Automatic Structuring of Indian Court Judgements


Automatic Structuring of judgments  focuses on segmenting a judgment into coherent chunks where each chunk
belongs to a particular rhetorical role.This segmentation can play an important role in tasks like
Summarization and act as building block for developing other legal  AI applications. 

###Need for structuring Indian court judgments
Indian court judgements are structured in form of paragraphs but there are no explicit sections mentioned in
the text. Judges have their own style of writing judgements which varies significantly by courts and case
types. Since the court judgements tend to be long, automatic structuring of the judgements will significantly
improve information retrieval and processing of the judgment.

###Data Collection
The data collection process was aimed at collecting sentence level rhetorical roles in Indian court judgements.
The data annotations were done voluntarily by Law students from multiple Indian law universities where each judgment sentence was 
classified  into one of the seven pre defined rhetorical role.For a detailed overview of the process,please refer to the paper

###Baseline
The baseline model was created using unified deep
learning architecture SciBERT-HSLN approach suggested by (Brack et al., 2021). SciBERT was replaced
with BERT BASE which are published by (Devlin et
al., 2018). Baseline model achieved micro f1 of 77.7.

###Evaluation metric
The evaluation metric used here is micro f1.


## Requirements

Python 3.8

To install the requirements,follow the instructions
```
pip install -r requirements.txt
```
To train the model on data other than the one provided,we will need to install the spacy transformers model
by following the steps below:

To install en_core_web_trf, run:
```
 python -m spacy download en_core_web_trf
```

## Data Download 


There are two  files:
- Training set http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
- Dev set in the distractor setting http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json



## Input Data Format

The top level structure of each JSON file is a list, where each entry represents a judgment-labels data point. Each data point is
a dict with the following keys:
- `id`: a unique id for this  data point. This is useful for evaluation.
- `annotations`:list of dict.The items in the dict are:
  - `result`a list of dictionaries containing sentence text and corresponding labels pair.The keys are:
    - `id`:unique id of each sentence
    - `value`:a dictionary with the following keys:
      - `start`:integer.starting index of the text
      - `end`:integer.end index of the text
      - `text`:string.The actual text of the sentence
      - `labels`:list.the labels that correspond to the text
- `data`: the actual text of the judgment.
- `meta`: a string.It tells about the category of the case(Criminal,Tax etc.)

##Model Training


###Data preparation for a different dataset

If you want to train the model on your own dataset,you will need to preprocess the data to convert it 
into the required json format.To do so,follow the following steps:

```
python infer_data_prep.py
```


### Preprocessing
  
  Preprocess the training and dev sets using:
  ```
  python tokenize_files.py
  ```
  
### Training
  
  Train the hsln  model
  ```
   python baseline_run.py 
  ```
  

## Inference

To run the inference,follow the following steps
```
python infer_new.py input_json_path output_json_path model_path

```
The output json will be written in the path provided as output_json_path.

## Prediction File Format

The prediction file format will be same as the training json with the following keys:
- `id`: a unique id for this  data point. This is useful for evaluation.
- `annotations`:list of dict.The items in the dict are:
  - `result`a list of dictionaries containing sentence text and corresponding labels pair.The keys are:
    - `id`:unique id of each sentence
    - `value`:a dictionary with the following keys:
      - `start`:integer.starting index of the text
      - `end`:integer.end index of the text
      - `text`:string.The actual text of the sentence
      - `labels`:list.the labels that correspond to the text
- `data`: the actual text of the judgment.
- `meta`: a string.It tells about the category of the case(Criminal,Tax etc.)
- 
## Model Submission and Test Set Evaluation

We use Codalab for test set evaluation. In the distractor setting, you must submit your code and provide a Docker environment. Your code will run on the test set.

## License
The automatic structuring dataset is distribued under the [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/legalcode) license.
The code is distribued under the Apache 2.0 license.

## References
