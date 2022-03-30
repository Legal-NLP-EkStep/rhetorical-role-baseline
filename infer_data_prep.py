import json
import os
import sys

import spacy
from tqdm import tqdm
from transformers import BertTokenizer

from data_prep import attach_short_sentence_boundries_to_next, seperate_and_clean_preamble, \
    get_spacy_nlp_pipeline_for_preamble
from data_prep import get_spacy_nlp_pipeline_for_indian_legal_text

spacy.prefer_gpu()


def split_into_sentences_tokenize_write(prediction_input_ls_format, custom_processed_data_path, nlp,
                                        hsln_format_txt_dirpath='datasets/pubmed-20k'):
    ########## This function accepts the input files in LS format, creates tokens and writes them with label as "NONE" to text file
    if not os.path.exists(hsln_format_txt_dirpath):
        os.makedirs(hsln_format_txt_dirpath)
    max_length = 10000
    output_json = []
    filename_sent_boundries = {}  ###### key is the filename and value is dict containing sentence spans {"abc.txt":{"sentence_span":[(1,10),(11,20),...]} , "pqr.txt":{...},...}
    for adjudicated_doc in tqdm(prediction_input_ls_format):

        doc_txt = adjudicated_doc['data']['text']
        preamble_text = adjudicated_doc['data']['preamble_text']
        judgement_text = adjudicated_doc['data']['judgement_text']
        file_name = adjudicated_doc['id']

        if filename_sent_boundries.get(file_name) is None:  ##### Ignore if the file is already present
            tokens = nlp.tokenizer(judgement_text)
            if len(tokens) > max_length:
                chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
                nlp_docs = [nlp(i.text) for i in tqdm(chunks, desc='Processing NLP chunks')]
                nlp_docs = [nlp(preamble_text)] + nlp_docs
                nlp_doc = spacy.tokens.Doc.from_docs(nlp_docs)
            else:
                nlp_preamble_doc = nlp(preamble_text)
                nlp_judgement_doc = nlp(judgement_text)
                nlp_doc = spacy.tokens.Doc.from_docs([nlp_preamble_doc, nlp_judgement_doc])
            doc_txt = nlp_doc.text
            sentence_boundries = [(sent.start_char, sent.end_char) for sent in nlp_doc.sents]
            revised_sentence_boundries = attach_short_sentence_boundries_to_next(sentence_boundries, doc_txt)
            adjudicated_doc['annotations'] = []
            adjudicated_doc['annotations'].append({})
            adjudicated_doc['annotations'][0]['result'] = []

            filename_sent_boundries[file_name] = {"sentence_span": []}
            for sentence_boundry in revised_sentence_boundries:
                sentence_txt = doc_txt[sentence_boundry[0]:sentence_boundry[1]]

                if sentence_txt.strip() != "":
                    sentence_txt = sentence_txt.replace("\r", "")
                    sent_data = {}
                    sent_data['value'] = {}
                    sent_data['value']['start'] = sentence_boundry[0]
                    sent_data['value']['end'] = sentence_boundry[1]
                    sent_data['value']['text'] = sentence_txt
                    sent_data['value']['labels'] = []
                    adjudicated_doc['annotations'][0]['result'].append(sent_data)
        output_json.append(adjudicated_doc)
    with open(custom_processed_data_path, 'w+') as f:
        json.dump(output_json, f)


if __name__ == "__main__":
    #[_, custom_data_path, custom_processed_data_path] = sys.argv
    # prediction_input_files_path = '/data/hsln_prediction/input_data_rgnlu.json' ###### in label studio format
    #     prediction_input_files_path = '/data/hsln_prediction/input_data_rgnlu.json'
    #     prediction_input_ls_format=[  {"id": 1,
    #   "data": { "text": "                                REPORTABLE\n\n IN THE SUPREME COURT OF INDIA\nCRIMINAL APPELLATE JURISDICTION\nCRIMINAL APPEAL NO. 1190 OF 2009\n\nState of Madhya Pradesh          ...Appellant\n\n                         Versus\n\nHarjeet Singh & Anr.             ...Respondents\n\n                                      J U D G M E N T\n INDU MALHOTRA, J.\n 1. The present Criminal Appeal has been filed by the State of Madhya Pradesh against the judgment and order dated 03.01.2006 passed by the Gwalior Bench of the Madhya Pradesh High Court, in Criminal Appeal No. 657/1998.\n           The Criminal Appeal was filed by the Respondents against their conviction under Section 307 of the Indian Penal Code (hereinafter referred to as \"Section 307\"). The High Court reduced the conviction of the Respondents from Section 307 to Section 324 of the Indian Penal Code (hereinafter referred to as \"Section 324\").\n2. The facts of the case, briefly stated, are as under: 2.1 The case of the Complainant  Sukhdev, as recorded in the F.I.R., is that on 12.11.1997 the ComplainantSukhdev along with his brothers  Balveer Yadav and Deshraj Yadav, had gone to the District Court, Ashok Nagar to attend the hearing of their case against Accused /Respondent No. 1  Harjeet Singh. After the hearing, at around noon, the Complainant  Sukhdev and his brothers crossed the road, and were standing in front of the Jail, when Ramji Lal Accused /Respondent No. 2 alongwith an unidentified assailant called Sardar caught hold of Balveer Yadav and Deshraj Yadav. The Accused /Respondent No. 1  Harjeet Singh grabbed the Complainant  Sukhdev, and stabbed him several times with a knife, inflicting blows on the chest, scapula, back, and hips.\n Accused /Respondent Nos. 1 and 2, alongwith Sardar ran away from the spot. The Complainant Sukhdev further stated that he would be able to identify Harjeet Singh, and the two assailants once he sees them.\n2." }
    #   },
    #   {"id": 2,
    #   "data": { "text": "2 Immediately after the assault on 12.11.1997, the Complainant  Sukhdev was admitted to the Civil Hospital, Ashok Nagar for treatment.\n2.3 The medical examination of the Complainant  Sukhdev was conducted by Dr. M. Bhagat  P.W.6 at the Civil Hospital, Ashok Nagar, which recorded the following injuries : (i) Stab Wound  3.5 x 1 cm  deep in the chest cavity, over the left side of the chest.\n        (ii) Spindle shaped incised wound  3 x 2 cm  muscle deep, present on the upper region of the right buttocks.\n        (iii) Stab Wound  2 x 1 cm  over subscapula region, left side. Bleeding was present.\n        (iv) Stab Wound  1 x 1 cm  over illeal region of hip, left side. Bleeding was present.\n The medical report further stated that the injuries were caused by a sharpedged, pointed object.\n 2.4 The Complainant  Sukhdev was referred to the District Hospital, Guna wherein XRay of his chest region was conducted by P.W. 8  Dr.\n       Raghuvanshi. The Report states that there was \"haziness in lungs, left side of chest, present due to trauma of chest\".\n               Dr. Raghuvanshi  P.W. 8 stated in his deposition that the lungs of the Complainant  Sukhdev suffered injury, which resulted in blood seeping in the lungs, leading to haziness in the X Ray image.\n 2.5 On 24.11.1997, the Accused /Respondent Nos. 1 and 2 were arrested by the Police. The weapon of offence i.e. the knife allegedly used by Accused /Respondent No. 1 was recovered from the bushes next to the bridge, on the statement given by Accused /Respondent No. 1.\n 2" }
    #   }
    # ]
    import re
    text = get_text_from_indiankanoon_url('https://indiankanoon.org/doc/82089984/')
    nlp_preamble = get_spacy_nlp_pipeline_for_preamble()
    preamble_text, preamble_end = seperate_and_clean_preamble(text, nlp_preamble)
    judgement_text = text[preamble_end:]
    judgement_text = re.sub(r'([^.\"\?])\n+ *', r'\1 ', judgement_text)
    # logger.info("Received text: '%s'", sentences)
    input_ls_format = [
        {'id': id, 'data': {'preamble_text': preamble_text, 'judgement_text': judgement_text,
                            'text': preamble_text + " " + judgement_text}}]

    #prediction_input_ls_format = json.load(open(custom_data_path))
    # prediction_input_ls_format = json.load(open(prediction_input_files_path))
    nlp = get_spacy_nlp_pipeline_for_indian_legal_text(model_name="en_core_web_trf",
                                                       disable=["attribute_ruler", "lemmatizer", 'ner'])
    BERT_VOCAB = "bert-base-uncased"
    BERT_MODEL = "bert-base-uncased"

    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

    MAX_DOCS = -1

    split_into_sentences_tokenize_write(input_ls_format, '/', nlp)
