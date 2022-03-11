
import streamlit as st
from annotated_text import annotated_text,annotation
import json

from api_inference_helper import get_text_from_indiankanoon_url, get_predicted_rhetorical_roles

inference_token = '6977a1d9ff0e4c5cb285f210ddb4ff49'
vm_ip_address = '35.232.131.254'


st.title("Judgment Summary & Rhetorical Role Identification App")
st.header("Identifies Rhetorical Role of each sentence of given Indian Court Judgement text. Also create summary for each Rheotrical Role.")


# annotated_text(
#     ("Vs. State of Punjab and others","PQR"),
#     #annotation("  \n  world!", "noun", style ={'color':"#8ef",'border':"1px dashed red"}),
#     ("  Vs.   ",""),
#     ("is", "verb", "#8ef"),
#     " some ",
#     ("annotated", "adj", "#faa"),
#     ("text", "noun", "#afa"),
#     " for those of ",
#     ("you", "pronoun", "#fea"),
#     " who ",
#     ("like", "verb", "#8ef"),
#     " this sort of ",
#     ("thing", "noun", "#afa"),
#     "."
# )
# 
indiankanoon_url = st.text_input('Paste IndianKanoon Link below and press Enter. E.g. https://indiankanoon.org/doc/41737003/','')

if indiankanoon_url!='':
    text = get_text_from_indiankanoon_url(indiankanoon_url)
    #output = get_predicted_rhetorical_roles(vm_ip_address, text,inference_token)
    output = json.load(open('/Users/astha/PycharmProjects/output.json'))
    sentences_with_rhetorical_roles = []
        ###### convert to streamlit format
    annotation_list = output[0]['annotations'][0]['result']
    for i,sentence_dict in enumerate(output[0]['annotations'][0]['result']):
            sentences_with_rhetorical_roles.append((sentence_dict['value']['text'],sentence_dict['value']['labels'][0]))
            if i < len(annotation_list) - 1 and annotation_list[i + 1]['value']['start'] > sentence_dict['value']['end']:
    
                sentences_with_rhetorical_roles.append(('<span style="white-space: pre-wrap">'+text[sentence_dict['value']['end']:annotation_list[i + 1]['value']['start']].replace('\n','&#10;').replace(' ','&#32;')+'</span>'))
         
    
    
        
    
    annotated_text(sentences_with_rhetorical_roles)
