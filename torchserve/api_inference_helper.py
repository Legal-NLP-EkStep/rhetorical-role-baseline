import json
import re
import urllib
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup as soup


def get_text_from_indiankanoon_url(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req, timeout=10).read()
    page_soup = soup(webpage, "html.parser")
    judgment_txt = page_soup.text
    ########### Remove the text before start of main judgment
    judgement_start_pattern = 'Free for one month and pay only if you like it.'
    if judgment_txt.find(judgement_start_pattern) != -1:
        judgment_txt = judgment_txt.split(judgement_start_pattern,1)[1]
    judgment_txt = judgment_txt.strip().split('\n',2)[2] ##### remove first two lines which tell court name & case

    ########### Remove the extra information added by IndianKanoon if it  exists
    judgment_txt = re.sub(r'^Equivalent citations\:.*\n','',judgment_txt)
    judgment_txt = re.sub(r'^Author\:.*\n', '', judgment_txt)
    judgment_txt = re.sub(r'^Bench\:.*\n', '', judgment_txt)
    return judgment_txt.strip()


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
