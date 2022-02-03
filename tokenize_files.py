"""Tokenizes the sentences with BertTokenizer as tokenisation costs some time.
"""
import sys
from transformers import BertTokenizer
import json
from sklearn.model_selection import train_test_split
BERT_VOCAB = "bert-base-uncased"
MAX_SEQ_LENGTH = 128


def write_in_hsln_format(input,hsln_format_txt_dirpath,tokenizer):


    final_string = ''
    filename_sent_boundries = {}
    for file in input:
        file_name=file['id']
        final_string = final_string + '###' + str(file_name) + "\n"
        filename_sent_boundries[file_name] = {"sentence_span": []}
        for annotation in file['annotations'][0]['result']:
            filename_sent_boundries[file_name]['sentence_span'].append([annotation['value']['start'],annotation['value']['end']])

            sentence_txt=annotation['value']['text']
            sentence_label = annotation['value']['labels'][0]
            sentence_txt = sentence_txt.replace("\r", "")
            if sentence_txt.strip() != "":
                sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=128)
                sent_tokens = [str(i) for i in sent_tokens]
                sent_tokens_txt = " ".join(sent_tokens)
                final_string = final_string + sentence_label + "\t" + sent_tokens_txt + "\n"
        final_string = final_string + "\n"
    with open(hsln_format_txt_dirpath , "w+") as file:
        file.write(final_string)




def tokenize():
    [_, train_input_json,dev_input_json,test_input_json] = sys.argv
    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)
    train_json_format = json.load(open(train_input_json))
    dev_json_format = json.load(open(dev_input_json))
    test_json_format = json.load(open(test_input_json))
  
    write_in_hsln_format(train_json_format,'datasets/pubmed-20k/train_scibert.txt',tokenizer)
    write_in_hsln_format(dev_json_format, 'datasets/pubmed-20k/dev_scibert.txt', tokenizer)
    write_in_hsln_format(test_json_format, 'datasets/pubmed-20k/test_scibert.txt', tokenizer)





tokenize()
