#!/usr/bin/env python
import json
import os.path
import sys

# as per the metadata file, input and output directories are the arguments
from sklearn.metrics import precision_recall_fscore_support

[_, ground_truth_path, input_json_path] = sys.argv

# unzipped submission data is always in the 'res' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
# submission_file_name = 'predictions.json'
# submission_dir = os.path.join(input_dir, 'res')
# submission_path = os.path.join(submission_dir, submission_file_name)
# if not os.path.exists(input_json_path):
#     message = "Expected submission file '{0}'"
#     sys.exit(message.format(input_json_path))

submission = json.load(open(input_json_path))
submission_id_label_map = {}
for doc in submission:
    for annotation in doc['annotations'][0]['result']:
        submission_id_label_map[annotation['id']] = annotation['value']['labels'][
            0]  ## consider only the first element in labels

# unzipped reference data is always in the 'ref' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
ground_truth = json.load(open(ground_truth_path))

ground_truth_labels = []
submission_lables = []
for doc in ground_truth:
    for annotation in doc['annotations'][0]['result']:
        ground_truth_label = annotation['value']['labels'][0]  ## consider only the first element in labels

        if submission_id_label_map.get(annotation['id']) is None:
            submission_label = "No_Prediction"
        else:
            submission_label = submission_id_label_map[annotation['id']]

        ground_truth_labels.append(ground_truth_label)
        submission_lables.append(submission_label)

precision, recall, f1, _ = precision_recall_fscore_support(ground_truth_labels, submission_lables,
                                                                             average='weighted')

# the scores for the leaderboard must be in a file named "scores.txt"
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
with open(os.path.join('result.json'), 'w') as output_file:
    json.dump({"Weighted-F1": f1}, output_file)

