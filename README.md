# BUILD: Benchmark for Understanding Indian Legal Documents


Indian Court Judgements have an inherent structure which is not explicitly mentioned in the judgement text. Assigning rhetorical roles to the sentences provides structure to the judgements. This is an important step which will act as building block for developing Legal AI solutions. We present benchmark for Rhetorical Role Prediction which include annotated data , evaluation methodology and baseline prediction model. This work is part of [OpenNyAI](https://opennyai.org/) mission which is funded by [EkStep Foundation](https://ekstep.org/).

## 1. What are Rhetorical Roles of Court Judgements?
Though there is no prescription for writing judgement,a judgement text follows an inherent structure. For example, a judgement text would begin with preamble, state facts of the case, courts analysis of the arguments from respondents and petitioners etc. Typical structure of an Indian court judgement is as shown below. The flow is not linear and these roles can appear in any sequence.
![Typical Structure of Indian Court Judgements](Rhetorical_Roles_Structure.png)
The detailed definitions of each of the rhetorical roles is specified in [table](/README.md#appendix). 
In this dataset, each of the sentence of the judgement text is marked with one rhetorical role. Task is to predict the rhetorical role of each sentence. 
This is sequential text classification because the rhetorical role of each of the sentence is not only dependent on the words of that sentence but also rhetorical roles of previous and next sentence. 
More details of Rhetorical Roles definitions with examples can be found in [MOOC](https://youtube.com/playlist?list=PL1z52lLL6eWnDnc3Wgfcu6neczrU3fFw0).

## 2. Data
The data collection process was aimed at collecting sentence level rhetorical roles in Indian court judgements.
The data annotations were done voluntarily by Law students from multiple Indian law universities where each sentence was 
classified  into one of the 13 pre-defined rhetorical roles.

### 2.1 Data Download 


There are two  files:
- [Training Data](https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/train.json)

- [Dev Data](https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/dev.json)


### 2.2 Input Data Format

The top level structure of each JSON file is a list, where each entry represents a judgement-labels data point. Each data point is
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
- `data`: the actual text of the judgement.
- `meta`: a string.It tells about the category of the case(Criminal,Tax etc.)


## 3. Inference of Baseline Model on dev data
The baseline model was created using unified deep
learning architecture SciBERT-HSLN approach suggested by (Brack et al., 2021). SciBERT was replaced
with BERT BASE which are published by (Devlin et
al., 2018). Baseline model achieved micro f1 of 77.7 on hiddent test data. 

### 3.1 Run Baseline Model on dev data
#### 3.1.1 Install Dependencies
Python 3.8

To install the requirements,follow the instructions
```
pip install -r requirements.txt
```
#### 3.1.2 Download pretrained model
[Model File](https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Model/model.pt)

#### 3.1.3 Run inference on dev data
 To run the inference on downloaded dev file , please run following command with appropriate paths
```
python infer_new.py dev_json_path output_json_path model_path

```

## 4. Submission & Evaluations
To preserve the integrity of test results, we do not release the test data to the public. Instead, we require you to submit your model so that we can run it on the test data for you.  We use Codalab for test data evaluation. Please refer to [Submission Guide](https://worksheets.codalab.org/worksheets/0xc791430db3c74588b87723255bd2d9db) for more details. 
The evaluation metric used here is micro f1.

## 5. Inference of Baseline Model on custom data

To train the model on data other than the one provided, we will need to split raw text into sentences. We use spacy transformer model for that. Install the spacy transformers model
by following the steps below:
To install the requirements,follow the instructions
```
pip install -r requirements.txt
```

To install en_core_web_trf, run:
```
 python -m spacy download en_core_web_trf
```


### 5.1 Data preparation for your own  data

If you want to train  the model on your own dataset and do the inference,you will need to preprocess the data to convert it 
into the required json format.To do so,follow the following steps:

The data prep file requires the data to be in the following format:
It should be a list of dict where each dict corresponds to a judgement.Each dict has the following keys:
- `id`: a unique id for this data point. This is useful for evaluation.
- `data`: dict with the key text which contains the actual text of the judgement.
For eg.
```
[  {"id": 1,
  "data": { "text": "input_text_1" } 
  },
  {"id": 2,
  "data": { "text": "input_text_2" } 
  },
   ....
]
```

To convert this into the format accepted by the model,run the data prep by:
```
python infer_data_prep.py custom_input.json custom_processed_input.json
```

### 5.2 Run Inference

To run the inference,follow the following steps
```
python infer_new.py custom_processed_input.json output_json_path model_path

```
The output json will be written in the path provided as output_json_path.

The prediction file format will be same as the training json with `labels` filled with predicted labels.


## 6. Training Baseline Model on train data
The training data is in the Input Data Format specified above.To train the model on your custom data,
please convert it into the required format.
For training baseline model on train data, follow steps

### 6.1 Preprocessing
  Once the data is in the input data format,the data needs to be tokenized and written in the
  particular folder.This can be done by:
  ```
  python tokenize_files.py train_json_path dev_json_path test_json_path
  ```
  
### 6.2 Run Training
  
  To train the HSLN model on the given data,we need to run the baseline_run.py.Model parameters like
max_epochs,num_batches etc. can be configured in the  baseline_run.py.To run the model on default parameters,
  ```
   python baseline_run.py 
  ```
  
## 7. Applications of Rhetorical Roles prediction
Automatic Structuring of Court judgements is foundation building block for creating other applications like summarization, automatic charge identification etc.  To try rhetorical rolewise summarization on custom judgement text using the baseline model, please refer to [Colab Notebook](https://colab.research.google.com/drive/1FRxgadvvMem8Z_Wtq_-CChk97X2DVdt2#scrollTo=NpdjaPELUkeJ).

## License
BUILD dataset is distribued under the [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/legalcode) license.
The code is distribued under the Apache 2.0 license.

## Acknowledgements
We thanks all the law student volunteers and coordinators associated with OpenNyAI for their contribution in data annotation.

## Appendix:
Rhetorical Roles Definititions
| Rhetorical Role                          | Rhetorical Roles (sentence level)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Preamble<br>(PREAMBLE)                   | A typical judgement would start with the court name, the details of parties, lawyers and judges' names, Headnotes. This section typically would end with a keyword like (JUDGEMENT or ORDER etc.)<br>Some supreme court cases also have HEADNOTES, ACTS section. They are also part of Preamble.                                                                                                                                                                                                                                                                                                               |
| Facts(FAC)                               | This refers to the chronology of events (but not judgement by lower court) that led to filing the case, and how the case evolved over time in the legal system (e.g., First Information Report at a police station, filing an appeal to the Magistrate, etc.)<br>Depositions and proceedings of current court<br>Summary of lower court proceedings                                                                                                                                                                                                                                                            |
| Ruling by Lower Court (RLC)              | Judgments given by the lower courts (Trial Court, High Court) based on which the present appeal was made (to the Supreme Court or high court). The verdict of the lower Court, Analysis & the ratio behind the judgement by the lower Court is annotated with this label.                                                                                                                                                                                                                                                                                                                                      |
| Issues (ISSUE)                           | Some judgements mention the key points on which the verdict needs to be delivered. Such Legal Questions Framed by the Court are ISSUES.<br>E.g. “he point emerge for determination is as follow:- (i) Whether on 06.08.2017 the accused persons in furtherance of their common intention intentionally caused the death of the deceased by assaulting him by means of axe ?”                                                                                                                                                                                                                                   |
| Argument by Petitioner (ARG\_PETITIONER) | Arguments by petitioners' lawyers. Precedent cases argued by petitioner lawyers fall under this but when court discusses them later then they belong to either the relied / not relied upon category.<br>E.g. “learned counsel for petitioner argued that …”                                                                                                                                                                                                                                                                                                                                                   |
| Argument by Respondent (ARG\_RESPONDENT) | Arguments by respondents lawyers. Precedent cases argued by respondent lawyers fall under this but when court discusses them later then they belong to either the relied / not relied upon category.<br>E.g. “learned counsel for the respondent argued that …”                                                                                                                                                                                                                                                                                                                                                |
| Analysis (ANALYSIS)                      | Courts discussion on the evidence,facts presented,prior cases and statutes. These are views of the court. Discussions on how the law is applicable or not applicable to current case. Observations(non binding) from court. It is the parent tag for 3 tags: PRE\_RLEIED, PRE\_NOT\_RELIED and STATUTE i.e. Every statement which belong to these 3 tags should also be marked as ANALYSIS<br><br>E.g. “Post Mortem Report establishes that .. “<br>E.g. “In view of the abovementioned findings, it is evident that the ingredients of Section 307 have been made out ….”                                     |
| Statute (STA)                            | Text in which the court discusses Established laws, which can come from a mixture of sources – Acts , Sections, Articles, Rules, Order, Notices, Notifications, Quotations directly from the bare act, and so on.<br>Statute will have both the tags Analysis + Statute<br><br>E.g. “Court had referred to Section 4 of the Code, which reads as under: "4. Trial of offences under the Indian Penal Code and other laws.-- (1) All offences under the Indian Penal Code (45 of 1860) shall be investigated, inquired into, tried, and otherwise dealt with according to the provisions hereinafter contained” |
| Precedent Relied (PRE\_RELIED)           | Sentences in which the court discusses prior case documents, discussions and decisions which were relied upon by the court for final decisions.<br>So Precedent will have both the tags Analysis + Precedent<br>E.g. This Court in Jage Ram v. State of Haryana3 held that: "12. For the purpose of conviction under Section 307 IPC, ….. “                                                                                                                                                                                                                                                                    |
| Precedent Not Relied (PRE\_NOT\_RELIED)  | Sentences in which the court discusses prior case documents, discussions and decisions which were not relied upon by the court for final decisions. It could be due to the fact that the situation in that case is not relevant to the current case.<br>E.g. This Court in Jage Ram v. State of Haryana3 held that: "12. For the purpose of conviction under Section 307 IPC, ….. “                                                                                                                                                                                                                            |
| Ratio of the decision (Ratio)            | Main Reason given for the application of any legal principle to the legal issue. This is the result of the analysis by the court.<br>This typically appears right before the final decision.<br>This is not the same as “Ratio Decidendi” taught in the Legal Academic curriculum.<br>E.g. “The finding that the sister concern is eligible for more deduction under Section 80HHC of the Act is based on mere surmise and conjectures also does not arise for consideration.”                                                                                                                                 |
| Ruling by Present Court (RPC)            | Final decision + conclusion + order of the Court following from the natural / logical outcome of the rationale<br>E.g. “In the result, we do not find any merit in this appeal. The same fails and is hereby dismissed.”                                                                                                                                                                                                                                                                                                                                                                                       |
| NONE                                     | If a sentence does not belong to any of the above categories<br>E.g. “We have considered the submissions made by learned counsel for the parties and have perused the record.”                                                                                                                                                                                                                                                                                                                                                                                                                                 |
