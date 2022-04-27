
# Abstractive Summarization of Legal docs using rhetorical roles

## 1. What is Abstractive Text Summarization?
Abstractive Text Summarization is the task of generating a short and concise summary that captures the salient ideas of the source text. Abstractive summarizers are so-called because they do not select sentences from the originally given text passage to create the summary. Instead, they produce a paraphrasing of the main contents of the given text, using a vocabulary set different from the original document.


## 2. Abstractive summarization using Rhetorical Roles
We wanted to see  how Rhetorical Roles could help generate better summaries. Towards this goal, we segmented the document in terms
of rhetorical roles, and each of the segments was passed
separately through the Legal Pegasus model to generate summaries. The final summary was obtained by
concatenating the summaries corresponding to each of
the rhetorical roles in the order they appear in the document.
For
this task we used tehd efault pre-trained Legal Pegasus model without any further training or finetuning.

## 3. What is Legal Pegasus?
[**Legal-Pegasus**](https://huggingface.co/nsi319/legal-pegasus) is a finetuned version of ([**google/pegasus-cnn_dailymail**](https://huggingface.co/google/pegasus-cnn_dailymail)) for the **legal domain**, trained to perform **abstractive summarization** task. The maximum length of input sequence is 1024 tokens.
This model was trained on [**sec-litigation-releases**](https://www.sec.gov/litigation/litreleases.htm) dataset consisting more than 2700 litigation releases and complaints.

## 4.Setup
 ```pip install -r requirements.txt```

## 5.Test data

The test data consisted of 50 documents randomly selected from the
Law Briefs dataset.To  get the raw data from lawbriefs, please request on this [email](adityagor282@gmail.com).
These legal documents come annotated with summaries
(abstractive summaries of the document) by expert editors.These summaries behaved as the gold standard summary for these judgments .The test data along with summaries with and without rhetorical role are [here](gs://indianlegalbert/OPEN_SOURCED_FILES/Abstractive_summarization/abstractive_summaries_data.csv)

## 6.Generating summaries on your own data


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

## Evaluation results
| Model                   | Rouge 1 | Rouge 2 | Rouge L |
|-------------------------|---------|---------|---------|
|Legal Pegasus with rr    | 0.56    | 0.36    | 0.48    |
|Legal Pegasus without rr |0.55     | 0.34    | 0.47    |
