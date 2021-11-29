from tqdm import tqdm
from transformers import BertTokenizer
import json
import os
from data_prep.summer_of_data.adjudicated_rhetorical_data_preparation import attach_short_sentence_boundries_to_next
from data_prep.utils import get_spacy_nlp_pipeline_for_indian_legal_text
import spacy
spacy.prefer_gpu()


def split_into_sentences_tokenize_write(prediction_input_ls_format,nlp):
    ########## This function accepts the input files in LS format, creates tokens and writes them with label as "NONE" to text file
    hsln_format_txt_dirpath ='datasets/pubmed-20k'
    if not os.path.exists(hsln_format_txt_dirpath):
        os.makedirs(hsln_format_txt_dirpath)
        
    final_string =''
    filename_sent_boundries = {} ###### key is the filename and value is dict containing sentence spans {"abc.txt":{"sentence_span":[(1,10),(11,20),...]} , "pqr.txt":{...},...}
    for adjudicated_doc in tqdm(prediction_input_ls_format):
        doc_txt = adjudicated_doc['data']['text']
        file_name = adjudicated_doc['meta']['file_name']
        if filename_sent_boundries.get(file_name) is None: ##### Ignore if the file is already present
            final_string=final_string+'###'+str(file_name)+ "\n"

            nlp_doc = nlp(doc_txt)
            sentence_boundries = [(sent.start_char,sent.end_char) for sent in  nlp_doc.sents]
            revised_sentence_boundries = attach_short_sentence_boundries_to_next(sentence_boundries,doc_txt)

            filename_sent_boundries[file_name] = {"sentence_span":[]}
            for sentence_boundry in revised_sentence_boundries:
                sentence_txt = doc_txt[sentence_boundry[0]:sentence_boundry[1]]
                sentence_txt = sentence_txt.replace("\r", "")
                if sentence_txt.strip() != "":
                    filename_sent_boundries[file_name]["sentence_span"].append(sentence_boundry)
                    sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=128)
                    sent_tokens = [str(i) for i in sent_tokens]
                    sent_tokens_txt = " ".join(sent_tokens)
                    final_string=final_string+"NONE"+"\t"+sent_tokens_txt+"\n"
            final_string = final_string +"\n"
        
    with open(hsln_format_txt_dirpath+'/train_scibert.txt',"w") as file:
        file.write(final_string)
    with open(hsln_format_txt_dirpath+'/test_scibert.txt',"w") as file:
        file.write(final_string)
    with open(hsln_format_txt_dirpath+'/dev_scibert.txt',"w") as file:
        file.write(final_string)
    with open(hsln_format_txt_dirpath+'/sentece_boundries.json','w') as json_file:
        json.dump(filename_sent_boundries,json_file)


if __name__=="__main__":

    prediction_input_files_path = '/data/hsln_prediction/input_data_rgnlu.json' ###### in label studio format

    prediction_input_ls_format = json.load(open(prediction_input_files_path))
    nlp = get_spacy_nlp_pipeline_for_indian_legal_text(model_name="en_core_web_trf",
                                                       disable=["attribute_ruler", "lemmatizer", 'ner'])
    BERT_VOCAB = "bert-base-uncased"
    BERT_MODEL = "bert-base-uncased"

    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

    MAX_DOCS = -1

    split_into_sentences_tokenize_write(prediction_input_ls_format, nlp)
