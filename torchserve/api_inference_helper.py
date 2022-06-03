import json
import re
import urllib
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup as soup,Tag



def get_text_from_indiankanoon_url(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    try:
        webpage = urlopen(req, timeout=10).read()
        page_soup = soup(webpage, "html.parser")

        preamble_tags = page_soup.find_all('pre')
        preamble_txt = ''.join([i.text for i in  preamble_tags if i.get('id') is not None and i['id'].startswith('pre_')])
        judgment_txt_tags = page_soup.find_all(['p','blockquote'])
        judgment_txt = ''
        for judgment_txt_tag in judgment_txt_tags:
            tag_txt=''
            if judgment_txt_tag.get('id') is not None and (judgment_txt_tag['id'].startswith('p_') or
                                                           judgment_txt_tag['id'].startswith('blockquote_')):
                for content in judgment_txt_tag.contents:
                    if isinstance(content,Tag):
                        if not(content.get('class') is not None and  'hidden_text' in content['class'] ):
                            tag_txt = tag_txt + content.text
                    else:
                        tag_txt = tag_txt + str(content)
                tag_txt = re.sub(r'\s+(?!\s*$)',' ',tag_txt) ###### replace the multiple spaces, newlines with space except for the ones at the end.
                tag_txt = re.sub(r'([.\"\?])\n',r'\1 \n\n',tag_txt) ###### add the extra new line for correct sentence breaking in spacy

                judgment_txt = judgment_txt + tag_txt
        judgment_txt = re.sub(r'\n{2,}', '\n\n', judgment_txt)
        judgment_txt = preamble_txt + judgment_txt

    except:
        judgment_txt=''

    return judgment_txt.strip()

#API Calling for Rhetorical role
def get_predicted_rhetorical_roles(ip, txt, inference_token):
    rr_api_url = f'http://{ip}:8080/predictions/RhetorcalRolePredictor/'
    body = {'text': txt, 'inference_token': inference_token}
    req = urllib.request.Request(rr_api_url)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')
    response = urllib.request.urlopen(req, jsondataasbytes).read()
    json_data = json.loads(response)
    return json_data

#API Calling for summary
def get_predicted_extractive_summary(ip, rhetorical_roles):
    rr_api_url = f'http://{ip}:8080/predictions/ExtractiveSummarizer/'
    body = rhetorical_roles
    req = urllib.request.Request(rr_api_url)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')
    response = urllib.request.urlopen(req, jsondataasbytes).read()
    json_data = json.loads(response)
    return json_data

#API Calling for NER
def get_legal_entities(ip, txt):
    rr_api_url = f'http://{ip}:8080/predictions/LegalNER/'
    body = {'txt': txt}
    req = urllib.request.Request(rr_api_url)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')
    response = urllib.request.urlopen(req, jsondataasbytes,timeout=520).read()
    json_data = json.loads(response)
    return json_data

def check_api_health(ip_address):
    api_url = f'http://{ip_address}:8080/ping'
    req = urllib.request.Request(api_url)
    try:
        status = json.load(urllib.request.urlopen(req))['status']
    except:
        status = 'UnHealthy'

    if status == 'Healthy':
        return True
    else:
        return False


def get_rhetorical_roles_from_indiankanoon_url(ik_url, inference_token, vm_ip_address):
    judgment_txt = get_text_from_indiankanoon_url(ik_url)
    predicted_rr = get_predicted_rhetorical_roles(vm_ip_address, judgment_txt, inference_token)
    return predicted_rr


if __name__ == "__main__":
    inference_token = ''
    vm_ip_address = ''
    if check_api_health(vm_ip_address):
        rhetorical_roles_output = get_rhetorical_roles_from_indiankanoon_url('https://indiankanoon.org/doc/103570654',
                                                                             inference_token=inference_token,
                                                                             vm_ip_address=vm_ip_address)
        extractive_summarizer_output = get_predicted_extractive_summary(vm_ip_address, rhetorical_roles_output)

        with open(
                "/Users/amantiwari/Projects/NLP_LEGAL/rhetorical-role-baseline/torchserve/predictor/rhetorical_roles_prediction_output.json",
                'w') as f:
            json.dump(rhetorical_roles_output, f, indent=4)

        with open(
                "/Users/amantiwari/Projects/NLP_LEGAL/rhetorical-role-baseline/torchserve/predictor/extractive_summarizer_prediction_output.json",
                'w') as f:
            json.dump(extractive_summarizer_output, f, indent=4)
