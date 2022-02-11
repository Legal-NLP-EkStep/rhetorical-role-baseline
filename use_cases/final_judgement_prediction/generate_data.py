import os

import pandas as pd

from infer_data_prep import split_into_sentences_tokenize_write, get_spacy_nlp_pipeline_for_indian_legal_text
from infer_new import *


def process_csv_for_rhetororical_role_prediction(csv_path='./Data/ILDC_Single/ILDC_single.csv'):
    df = pd.read_csv(csv_path)
    f_names = df['name'].to_list()
    custom_data = []
    for i, text in enumerate(df['text'].to_list()):
        custom_data.append({"id": f_names[i], "data": {"text": text}})
    nlp = get_spacy_nlp_pipeline_for_indian_legal_text(model_name="en_core_web_trf",
                                                       disable=["attribute_ruler", "lemmatizer", 'ner'])
    path = os.path.abspath('ILDC_rr_input.json')
    split_into_sentences_tokenize_write(custom_data, path, nlp)
    return path


def generate_rhetorical_role_predictions(input_dir='ILDC_rr_input.json', model_path='model.pt',
                                         prediction_output_json_path='ILDC_rr_output.json'):
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

    hsln_format_txt_dirpath = 'datasets/pubmed-20k'
    write_in_hsln_format(input_dir, hsln_format_txt_dirpath, tokenizer)
    filename_sent_boundries = json.load(open(hsln_format_txt_dirpath + '/sentece_boundries.json'))
    predictions = infer(model_path, MAX_DOCS, prediction_output_json_path)

    ##### write the output in format needed by revision script
    for doc_name, predicted_labels in zip(predictions['doc_names'], predictions['docwise_y_predicted']):
        filename_sent_boundries[doc_name]['pred_labels'] = predicted_labels
    with open(input_dir, 'r') as f:
        input = json.load(f)
    for file in input:
        id = str(file['id'])
        pred_id = predictions['doc_names'].index(id)
        pred_labels = predictions['docwise_y_predicted']
        annotations = file['annotations']
        for i, label in enumerate(annotations[0]['result']):
            label['value']['labels'] = [pred_labels[pred_id][i]]

    with open(prediction_output_json_path, 'w') as file:
        json.dump(input, file)


def generate_final_csv(csv_path='./Data/ILDC_Single/ILDC_single.csv',
                       prediction_output_json_path='ILDC_rr_output.json'):
    data = json.load(open(prediction_output_json_path))
    df = pd.read_csv(csv_path)

    df_f_name = df['name'].to_list()

    rhetorical_roles = ['PREAMBLE', 'NONE', 'FAC', 'ISSUE', 'ARG_RESPONDENT', 'ARG_PETITIONER', 'ANALYSIS',
                        'PRE_RELIED', 'PRE_NOT_RELIED', 'STA', 'RLC', 'RPC', 'RATIO']
    new_colomns = {}
    for key in rhetorical_roles:
        new_colomns[key] = []

    for file_name in df_f_name:
        temp = {}
        for key in rhetorical_roles:
            temp[key] = []
        for doc in data:
            if doc['id'] == file_name:
                for values in doc['annotations'][0]['result']:
                    temp[values['value']['labels'][0]].append(values['value']['text'])
        for key in rhetorical_roles:
            new_colomns[key].append('\n'.join(temp[key]))
    for key in rhetorical_roles:
        df[key] = new_colomns[key]

    input_text = []
    for text, ratio, rpc in zip(df['text'].to_list(), df['RATIO'].to_list(), df['RPC'].to_list()):
        if not ratio:
            ratio = ''
        else:
            ratio = ratio.split('\n')
        if not rpc:
            rpc = ''
        else:
            rpc = rpc.split('\n')
        for ratio_line in ratio:
            text = text.replace(ratio_line, '')
        for rpc_line in rpc:
            text = text.replace(rpc_line, '')
        input_text.append(text)
    df['input_text'] = input_text
    df.to_csv("ILDC_single_new_with_rhetorical_roles", index=False)


if __name__ == '__main__':
    IDLC_SINGLE_INPUT_PATH, RHETORICAL_ROLE_MODEL_PATH = sys.argv[1], sys.argv[2]

    input_dir = process_csv_for_rhetororical_role_prediction(csv_path=IDLC_SINGLE_INPUT_PATH)
    generate_rhetorical_role_predictions(input_dir=input_dir, model_path=RHETORICAL_ROLE_MODEL_PATH,
                                         prediction_output_json_path='ILDC_rr_output.json')

    generate_final_csv(csv_path=IDLC_SINGLE_INPUT_PATH, prediction_output_json_path='ILDC_rr_output.json')
