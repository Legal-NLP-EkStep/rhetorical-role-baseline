from rouge_score import rouge_scorer
from tqdm import tqdm
import json
from SummaryGeneration import SummaryGeneration
import re
import sys
import pandas as pd


def concatenate_text_by_rhetorical_role(annotation_list):
    #### concatenate text for each rhetorical role
    rhetorical_rolewise_text = {} # keys are rhetorical roles and value is concatenated text
    for annotation in annotation_list:
      sent_role = annotation['value']['labels'][0]
      sent_txt = annotation['value']['text'].strip()
      if rhetorical_rolewise_text.get(sent_role) is None:
        rhetorical_rolewise_text[sent_role] = sent_txt
      else:
        if not rhetorical_rolewise_text[sent_role].endswith('.'):
          rhetorical_rolewise_text[sent_role] = rhetorical_rolewise_text[sent_role] + '. '+ sent_txt
        else:
          rhetorical_rolewise_text[sent_role] = rhetorical_rolewise_text[sent_role] + ' '+ sent_txt
    return rhetorical_rolewise_text

def split_preamble_judgement(judgment_txt):
    ###### seperates the preamble and judgement text for all courts. It removes the new lines in between  the sentences.  returns 2 texts
    preamble_end = remove_unwanted_text(judgment_txt)
    preamble_removed_txt = judgment_txt[preamble_end:]
    preamble_txt = judgment_txt[:preamble_end]

    ####### remove the new lines which are not after dot or ?. Assumption is that theses would be in between sentence
    preamble_removed_txt = re.sub(r'([^.\"\?])\n+ *', r'\1 ',
                                  preamble_removed_txt)
    return  preamble_txt,preamble_removed_txt

def remove_unwanted_text(text):
    '''Looks for pattern  which typically starts the main text of jugement.
    The text before this pattern contains metadata like name of paries, judges and hence removed'''
    pos_list = []
    len = 0
    pos = 0
    pos_list.append(text.find("JUDGMENT & ORDER"))
    pos_list.append(text.find("J U D G M E N T"))
    pos_list.append(text.find("JUDGMENT"))
    pos_list.append(text.find("O R D E R"))
    pos_list.append(text.find("ORDER"))

    for i, p in enumerate(pos_list):

        if p != -1:
            if i == 0:
                len = 16
            elif i == 1:
                len = 15
            elif i == 2:
                len = 8
            elif i == 3:
                len = 9
            elif i == 4:
                len = 5
            pos = p + len
            break

    return pos


def create_summary_with_rr(predictions):
    final_summaries_rr = []
    final_summaries_normal = []
    rheotrical_roles_to_summarize = ['FAC', 'RLC', 'ISSUE', 'ARG_PETITIONER', 'ARG_RESPONDENT', 'ANALYSIS',
                                     'PRECEDENT_RELIED', 'PRECEDENT_NOT_RELIED', 'STA', 'RATIO', 'RPC']
    for index, prediction in enumerate(predictions):

        inp = prediction['data']
        preamble_txt, preamble_removed_txt = split_preamble_judgement(inp['text'])
        cleaned_text = preamble_txt + preamble_removed_txt

        sent_tags = []
        id = prediction['id']
        text = prediction['data']['text']
        summ_dict = {}
        summ_dict['id'] = id
        annotation_list = prediction['annotations'][0]['result']
        for i, annotation in enumerate(annotation_list):
            sent_tags.append((annotation['value']['text'], annotation['value']['labels'][0]))
            if i < len(annotation_list) - 1 and annotation_list[i + 1]['value']['start'] > annotation['value']['end']:
                sent_tags.append(
                    (cleaned_text[annotation['value']['end']:annotation_list[i + 1]['value']['start']], None))

        ##### create summary without rhetorical role
        normal_summ_dict={}
        normal_summ_dict['id'] = id

        normal_generated_summaries = legal_summarizer.generate(cleaned_text, token_max_length=1024)
        normal_summ_dict['summary_normal'] = normal_generated_summaries[0]['level_1_summary']
        final_summaries_normal.append(normal_summ_dict)

        ##### create summary for each rhetorical role

        rhetorical_rolewise_text = concatenate_text_by_rhetorical_role(annotation_list)

        generated_summaries_dict = {}  ##### key is rhetorical role and value is dict of {"summary":'fdsf...', "is_summarized":True}
        for rhet_role, rhet_text in rhetorical_rolewise_text.items():
            ######## create summary only if the text is long enough
            if len(rhet_text) > 512:
                generated_summaries = legal_summarizer.generate(rhet_text, token_max_length=1024)
                generated_summaries_dict[rhet_role] = {"summary": generated_summaries[0]["level_1_summary"],
                                                       "is_summarized": True}
            else:
                generated_summaries_dict[rhet_role] = {"summary": rhet_text, "is_summarized": False}
        final_summary = ''

        for rhetorical_role in rheotrical_roles_to_summarize:
            if rhetorical_role == 'NONE':
                continue
            if generated_summaries_dict.get(rhetorical_role) is not None:
                final_summary = final_summary + generated_summaries_dict[rhetorical_role]['summary']

        summ_dict['summary_rr'] = final_summary

        final_summaries_rr.append(summ_dict)

    return final_summaries_rr,final_summaries_normal



def rouge(label_sent: list, pred_sent: list) -> dict:
    """Approximate ROUGE scores, always run externally for final scores."""
    individual_rouges=[]
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"],
        use_stemmer=True)
    r1, r2, rl = 0.0, 0.0, 0.0
    for ls, ps in tqdm(zip(label_sent, pred_sent), desc=f"Calculating scores on {len(label_sent)} labels",
                       total=len(label_sent)):
        score = scorer.score(ls, ps)
        individual_rouges.append( {"eval_rouge-1":  score["rouge1"],
              "eval_rouge-2":  score["rouge2"],
              "eval_rouge-L":score["rougeLsum"]})
        r1 += score["rouge1"].fmeasure
        r2 += score["rouge2"].fmeasure
        rl += score["rougeLsum"].fmeasure
    result = {"eval_rouge-1": r1 / len(label_sent),
              "eval_rouge-2": r2 / len(label_sent),
              "eval_rouge-L": rl / len(label_sent)}
    return result


if __name__ == "__main__":
    arguments= sys.argv
    predictions=arguments[1]
    output_path=arguments[2]

    legal_summarizer = SummaryGeneration(model="nsi319/legal-pegasus", tokenizer="nsi319/legal-pegasus")
    rr_summaries,normal_summaries = create_summary_with_rr(json.load(open(predictions,'r')))
    json.dump(rr_summaries,open(output_path+'/rr_summaries.json','w+'))
    json.dump(normal_summaries, open(output_path + '/normal_summaries.json', 'w+'))
  
    if len(arguments)==4:
        summary_judgment_mapping=json.load(open(arguments[3],'r'))
        original_summary=[]
        predicted_summary_rr=[]
        predicted_summary_normal=[]

    
        for summary_judgment in summary_judgment_mapping:
            id=summary_judgment['id']
            original_summary.append(summary_judgment['original_summary'])
            rr_summary=[summary['summary_rr'] for summary in rr_summaries if summary['id']==id][0]
            normal_summary=[summary['summary_normal'] for summary in normal_summaries if summary['id']==id][0]
            predicted_summary_rr.append(rr_summary)
            predicted_summary_normal.append(normal_summary)
    
        rouge_score_rr=rouge(original_summary,predicted_summary_rr)
        rouge_score_normal=rouge(original_summary,predicted_summary_normal)
        print('Rouge_score for Rhetorical Role summaries is :'+str(rouge_score_rr))
        print('Rouge_score for normal summaries is :' + str(rouge_score_normal))



