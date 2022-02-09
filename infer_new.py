import models
import json
import sys

import torch
from transformers import BertTokenizer

import models
from eval import eval_model
from models import BertHSLN
from task import pubmed_task
from utils import get_device


def create_task(create_func):
    return create_func(train_batch_size=config["batch_size"], max_docs=MAX_DOCS)


def infer(model_path, max_docs, prediction_output_json_path, device):
    ######### This function loads the model from given model path and predefined data. It then predicts the rhetorical roles and returns
    task = create_task(pubmed_task)
    model = getattr(models, config["model"])(config, [task]).to(device)
    model.load_state_dict(torch.load(model_path))

    folds = task.get_folds()
    test_batches = folds[0].test
    metrics, confusion, labels_dict, class_report = eval_model(model, test_batches, device, task)

    return labels_dict
def write_in_hsln_format(input_json,hsln_format_txt_dirpath,tokenizer):

    #tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)
    json_format = json.load(open(input_json))
    final_string = ''
    filename_sent_boundries = {}
    for file in json_format:
        file_name=file['id']
        final_string = final_string + '###' + str(file_name) + "\n"
        filename_sent_boundries[file_name] = {"sentence_span": []}
        for annotation in file['annotations'][0]['result']:
            filename_sent_boundries[file_name]['sentence_span'].append([annotation['value']['start'],annotation['value']['end']])

            sentence_txt=annotation['value']['text']
            sentence_txt = sentence_txt.replace("\r", "")
            if sentence_txt.strip() != "":
                sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=128)
                sent_tokens = [str(i) for i in sent_tokens]
                sent_tokens_txt = " ".join(sent_tokens)
                final_string = final_string + "NONE" + "\t" + sent_tokens_txt + "\n"
        final_string = final_string + "\n"

    with open(hsln_format_txt_dirpath + '/test_scibert.txt', "w+") as file:
        file.write(final_string)

    with open(hsln_format_txt_dirpath + '/train_scibert.txt', "w+") as file:
        file.write(final_string)
    with open(hsln_format_txt_dirpath + '/dev_scibert.txt', "w+") as file:
        file.write(final_string)
    with open(hsln_format_txt_dirpath + '/sentece_boundries.json', 'w+') as json_file:
        json.dump(filename_sent_boundries, json_file)

    return filename_sent_boundries

if __name__=="__main__":
    [_,input_dir, prediction_output_json_path, model_path] = sys.argv

    BERT_VOCAB = "bert-base-uncased"
    BERT_MODEL = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

    config = {
        "bert_model": BERT_MODEL,
        "bert_trainable": False,
        "model": BertHSLN.__name__,
        "cacheable_tasks": [],

        "dropout": 0.5,
        "word_lstm_hs": 758,
        "att_pooling_dim_ctx": 200,
        "att_pooling_num_ctx": 15,

        "lr": 3e-05,
        "lr_epoch_decay": 0.9,
        "batch_size": 32,
        "max_seq_length": 128,
        "max_epochs": 40,
        "early_stopping": 5,

    }


    MAX_DOCS = -1
    device = get_device(0)

    
    hsln_format_txt_dirpath ='datasets/pubmed-20k'
    write_in_hsln_format(input_dir,hsln_format_txt_dirpath,tokenizer)
    filename_sent_boundries = json.load(open(hsln_format_txt_dirpath + '/sentece_boundries.json'))
    predictions = infer(model_path, MAX_DOCS, prediction_output_json_path, device)
    
    ##### write the output in format needed by revision script
    for doc_name,predicted_labels in zip(predictions['doc_names'],predictions['docwise_y_predicted']):
        filename_sent_boundries[doc_name]['pred_labels'] = predicted_labels
    with open(input_dir,'r') as f:
        input=json.load(f)
    for file in input:
        id=str(file['id'])
        pred_id=predictions['doc_names'].index(id)
        pred_labels=predictions['docwise_y_predicted']
        annotations=file['annotations']
        for i,label in enumerate(annotations[0]['result']):

            label['value']['labels']=[pred_labels[pred_id][i]]

    with open(prediction_output_json_path,'w') as file:
        json.dump(input,file)
