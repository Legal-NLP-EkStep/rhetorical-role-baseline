import json
import urllib
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup as soup


def get_text_from_indiankanoon_url(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req, timeout=10).read()
    page_soup = soup(webpage, "html.parser")
    return page_soup.text.strip()


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


def get_rhetorical_roles_from_indiankanoon_url(ik_url, inference_token, vm_ip_address):
    judgment_txt = get_text_from_indiankanoon_url(ik_url)
    predicted_rr = get_predicted_rhetorical_roles(judgment_txt, inference_token, ip=vm_ip_address)
    return predicted_rr


if __name__ == "__main__":
    inference_token = '6977a1d9ff0e4c5cb285f210ddb4ff49'
    vm_ip_address = ''
    output = get_rhetorical_roles_from_indiankanoon_url('https://indiankanoon.org/doc/137175626/',
                                                        inference_token=inference_token, vm_ip_address=vm_ip_address)
    with open("output.json", 'w') as f:
        json.dump(output, f)
