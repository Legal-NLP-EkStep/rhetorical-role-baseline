# CJPE (Court Judgment Prediction and Explanation)

This code base contains files for predicting and explaining the court judgments based on legal judgment texts.

## CJPE data

For dataset you can contact authors of ACL-IJCNLP 2021 paper "ILDC for CJPE: Indian Legal Documents Corpus for Court
Judgment Prediction and Explanation" or visit their [github repo](https://github.com/Exploration-Lab/CJPE).

## Processing raw data

We process each judgement and pass them through our rhetorical role model to prove the effectiveness of using specific
rhetorical role for judgement prediction rather than using whole judgement or part of judgement. To process the raw CSV
file and create new processed data with text separated based on rhetorical roles.

Run:

```
python generate_data.py ./Data/ILDC_Single/ILDC_single.csv rhetorical_role_model.pt
```

Note: Do change the path for rhetorical role model and raw data file

## Generating results

Use the XLNet_on_single.ipynb notebook for running the code

### For baseline

Set variable ```COLOMN_NAME``` in cell 2 to ```input_text```

```
COLOMN_NAME = 'input_text'
```

### For getting results on rhetorical role ```ANALYSIS```

Set variable ```COLOMN_NAME``` in cell 2 to ```ANALYSIS```

```
COLOMN_NAME = 'ANALYSIS'
```