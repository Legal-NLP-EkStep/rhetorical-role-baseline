
## PEGASUS for legal document summarization
**legal-pegasus** is a finetuned version of ([**google/pegasus-cnn_dailymail**](https://huggingface.co/google/pegasus-cnn_dailymail)) for the **legal domain**, trained to perform **abstractive summarization** task. The maximum length of input sequence is 1024 tokens.

##Setup
 ```pip install -r requirements.txt```

## Training data

This model was trained on [**sec-litigation-releases**](https://www.sec.gov/litigation/litreleases.htm) dataset consisting more than 2700 litigation releases and complaints.

## Test data

The test data consisted of 50 documents randomly selected from the
Law Briefs dataset.These legal documents come annotated with summaries
(abstractive summaries of the document) by expert editors.These summaries behaved as the gold standard summary for these judgments .

##Generating summaries on your own data


 To generate summaries,please create predictions file using the rhetorical role model.Then 
 run the following command with appropriate paths
```
python generate_summaries.py --predictions_path --output_path
```

To generate summaries and calculate rouge score with gold standard summaries:
```
python generate_summaries.py --predictions_path --output_path --judgment_summary_mapping.json
```
where:

```<predictions_path>: path to the json file which is output of the Rhetorical role model```

```<output_path>:path where the generated summaries json is to be saved```

```<judgment_summary_mapping>:path to the json consisting of judgment and gold standard summaries```

##Evaluation results
