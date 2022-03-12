import html
import streamlit as st
from annotated_text import annotated_text,annotation
import json
import re
from api_inference_helper import get_text_from_indiankanoon_url, get_predicted_rhetorical_roles

inference_token = '6977a1d9ff0e4c5cb285f210ddb4ff49'
vm_ip_address = '35.232.131.254'


st.title("Judgment Summary & Rhetorical Role Identification App")
st.header("Identifies Rhetorical Role of each sentence of given Indian Court Judgement text. Also create summary for each Rheotrical Role.")


indiankanoon_url = st.text_input('Paste IndianKanoon Link below and press Enter. E.g. https://indiankanoon.org/doc/41737003/','https://indiankanoon.org/doc/41737003/')

def convert_text_to_html_entities(txt):
    return txt.replace('\n','&#10;').replace('.','&#46;').replace('*','&#42;')

color_map={"PREAMBLE": "#009757", "FAC": "#B82D00","RLC":"#01A09E",
"ISSUE":"#155489","ARG_PETITIONER":"#3835ED","ARG_RESPONDENT":"#7A22D8","STA":"Thistle","ANALYSIS":"#8B8B0F","PRE_RELIED":"YellowGreen","PRE_NOT_RELIED":"#A01AC4",
"RATIO":"Orange","RPC":"#C10202","NONE":"#02A2A4"}

if indiankanoon_url!='':
    text = get_text_from_indiankanoon_url(indiankanoon_url)
    #output = get_predicted_rhetorical_roles(vm_ip_address, text,inference_token)
    output = json.load(open('output.json'))
    sentences_with_rhetorical_roles = []

    ###### convert to streamlit format
    annotation_list = output[0]['annotations'][0]['result']
    anno_txt=''
    prev_label= None
    for i,sentence_dict in enumerate(output[0]['annotations'][0]['result']):
        sent_txt = convert_text_to_html_entities(sentence_dict['value']['text'])
        sent_rr = sentence_dict['value']['labels'][0]
        if sent_rr == prev_label:
            anno_txt = anno_txt + sent_txt
        else:
            ###### there is change of rhetorical role
            if anno_txt!='':
                sentences_with_rhetorical_roles.append((anno_txt,prev_label,color_map.get(prev_label)))
            prev_label = sent_rr
            anno_txt = sent_txt

        ########## add the text which is in between 2 annotations
        if i < len(annotation_list) - 1 and annotation_list[i + 1]['value']['start'] > sentence_dict['value']['end']:
            unmarked_sent = text[sentence_dict['value']['end']:annotation_list[i + 1]['value']['start']]
            unmarked_sent = convert_text_to_html_entities(unmarked_sent) ######## as per html requirements
            #unmarked_sent = '<span style="white-space: pre-wrap">'+ unmarked_sent +'</span>'
            anno_txt = anno_txt + unmarked_sent

    
    annotated_text(sentences_with_rhetorical_roles)
