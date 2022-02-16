import copy
import json
import sys

from transformers import BertTokenizer


def convert_to_bertsum_format(summaries_judgments_mapping):
    bertsum_format_list = []
    from spacy.lang.en import English
    nlp = English()
    tokenizer = nlp.tokenizer
    min_src_nsents = 5
    min_tgt_ntokens = 100
    max_bert_tokens_per_chunk = 512

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    for doc in summaries_judgments_mapping:

        source_chunk_id = 0
        chunk_bert_token_length = 0
        doc_data = {"src": [], 'tgt': [], "src_rhetorical_roles": [],
                    'source_filename': doc['id'], 'src_chunk_id': source_chunk_id}
        doc_data_list = []

        tgt_len = 0
        for sent_dict in doc['annotations'][0]['result']:
            sent_tokens = [token.text for token in tokenizer(sent_dict['value']['text'])]
            bert_sent_tokens = bert_tokenizer.tokenize(sent_dict['value']['text'])
            if sent_dict['value']['labels'] not in ["PREAMBLE", "NONE"]:
                if chunk_bert_token_length + len(bert_sent_tokens) >= max_bert_tokens_per_chunk:
                    doc_data_list.append(copy.deepcopy(doc_data))
                    source_chunk_id += 1
                    chunk_bert_token_length = 0
                    doc_data = {"src": [], 'tgt': [], "src_rhetorical_roles": [],
                                'source_filename': doc['id'],
                                'src_chunk_id': source_chunk_id}

                doc_data['src'].append(sent_tokens)
                doc_data['src_rhetorical_roles'].append(sent_dict['value']['labels'])
                chunk_bert_token_length += len(bert_sent_tokens)
                doc_data['tgt'].append([])

        ####### append the last chunk
        doc_data_list.append(copy.deepcopy(doc_data))

        if len(doc['annotations'][0]['result']) >= min_src_nsents:
            bertsum_format_list.extend(doc_data_list)
    return bertsum_format_list



if __name__ == "__main__":
    arguments = sys.argv
    predictions = arguments[1]
    output_path = arguments[2]
    bertsum_format_list = convert_to_bertsum_format(predictions)
    json.dump(bertsum_format_list,open(output_path+'bertsum_format.json','w+'))