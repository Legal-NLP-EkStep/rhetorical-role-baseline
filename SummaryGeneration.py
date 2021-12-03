import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def logger(statement):
    print(statement)


class SummaryGeneration:
    def __init__(self, model="nsi319/legal-pegasus", tokenizer="nsi319/legal-pegasus"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.model = model.to(self.device)
        self.warn1 = True
        self.warn2 = True

    def _tokenise_to_tensor(self, text: str, max_length=1024, overlap=0, overlap_all=False, overlap_last=False,
                            cont=False) -> list:
        """Tokenize input text to tensor input
        This function allows overlap on all chunks or on only last chunk of tokens by utilizing the parameters"""
        overlap_len_last = 0
        if self.tokenizer.model_max_length and max_length != self.tokenizer.model_max_length and self.warn1:
            logger(
                f"token max length not equal to model max length({self.tokenizer.model_max_length}), this length will be used for tokenization and can affect model capability")
            self.warn1 = False
        assert 0 <= overlap <= max_length, "Overlap cannot be negative or greater than max_length"
        if overlap_all:
            overlap_last = False
        if overlap_all or overlap_last:
            if overlap == 0:
                logger("Overlap is 0, please change")
        if overlap_last:
            overlap_len_last = overlap
            overlap = 0

        tokens = self.tokenizer.tokenize(text)
        tokens_list = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length - overlap)]
        text_list = [self.tokenizer.convert_tokens_to_string(i) for i in tokens_list]

        if not cont and len(tokens_list[-1]) < 512 and self.warn2:
            logger("Last split of tokens is less than 512, suggested doing overlap and setting token_overlap_last=True")
            self.warn2 = False
        if overlap_last:
            sub = tokens_list[-2][-overlap_len_last:] + tokens_list[-1]
            text_list[-1] = self.tokenizer.convert_tokens_to_string(sub)
        input_tokenized = [
            self.tokenizer.encode(text, return_tensors='pt', max_length=max_length, truncation=True).to(self.device)
            for text in text_list]
        return input_tokenized

    def _generate_summary(self, tokenized_input: list, summary_beams: int, summary_ngram: int,
                          summary_length_penalty: float,
                          summary_min_length: int, summary_max_length: int) -> str:
        """This funtion is used to generate summaries for tokenized inputs"""
        summary = []
        for each_tokenized_input in tokenized_input:
            each_tokenized_input = each_tokenized_input.to(self.device)
            summary_ids = self.model.generate(each_tokenized_input,
                                              num_beams=summary_beams,
                                              no_repeat_ngram_size=summary_ngram,
                                              length_penalty=summary_length_penalty,
                                              min_length=summary_min_length,
                                              max_length=summary_max_length,
                                              early_stopping=True)
            summary.append(
                [self.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in
                 summary_ids][0])
        return "\n".join(summary)

    def generate(self, input: list or str, summary_beams=10, summary_ngram=4, summary_length_penalty=2.0,
                 summary_min_length=150,
                 summary_max_length=512, token_max_length=1024, token_overlap=0, token_overlap_all=False,
                 token_overlap_last=False, level_2=False) -> list:
        """This function accepts string or list of string and then generates summaries on each string"""
        if type(input) == str:
            input = [input]

        assert type(input) == list, "Allowed input type is string or list of strings"
        assert all(isinstance(n, str) for n in input), "Allowed input type is string or list of strings"

        output = []

        for text in tqdm(input, desc="Generating summaries"):
            tokenized_input = self._tokenise_to_tensor(text, token_max_length, token_overlap, token_overlap_all,
                                                       token_overlap_last)
            level_1_summary = self._generate_summary(tokenized_input, summary_beams, summary_ngram,
                                                     summary_length_penalty, summary_min_length, summary_max_length)
            if level_2:
                tokenized_input = self._tokenise_to_tensor(level_1_summary, token_max_length, token_overlap,
                                                           token_overlap_all,
                                                           token_overlap_last, cont=True)
                level_2_summary = self._generate_summary(tokenized_input, summary_beams, summary_ngram,
                                                         summary_length_penalty, summary_min_length, summary_max_length)
                final = {"level_1_summary": level_1_summary, "level_2_summary": level_2_summary}
            else:
                final = {"level_1_summary": level_1_summary}
            output.append(final)

        return output


if __name__ == "__main__":
    import json
    from pathlib import Path
    from rouge_metric import rouge

    json_paths = [str(i) for i in list(Path("./test_data").glob("*.json"))]
    judgements = []
    labels = []
    for path in tqdm(json_paths, desc="Processing Json's"):
        judgement = ''
        label = ''
        data = json.loads(open(str(path)).read())
        for pair in data["summary_text_pairs"]:
            judgement += pair["text"]
            label += pair["summary"]
        judgements.append(judgement)
        labels.append(label)

    legal_summarizer = SummaryGeneration(model="nsi319/legal-pegasus", tokenizer="nsi319/legal-pegasus")
    generated_summaries = legal_summarizer.generate(judgements, token_max_length=1024)
    generated_summaries = [summary["level_1_summary"] for summary in generated_summaries]
    rouge_scores = rouge(labels, generated_summaries)
    logger(
        f"Legal Pegasus socores:\nRouge-1 - {rouge_scores['eval_rouge-1']}\nRouge-2 - {rouge_scores['eval_rouge-2']}\nRouge-L - {rouge_scores['eval_rouge-L']}\n")

    bigbird_summarizer = SummaryGeneration(model="google/bigbird-pegasus-large-arxiv",
                                           tokenizer="google/bigbird-pegasus-large-arxiv")
    generated_summaries = bigbird_summarizer.generate(judgements, token_max_length=4096)
    generated_summaries = [summary["level_1_summary"] for summary in generated_summaries]
    rouge_scores = rouge(labels, generated_summaries)
    logger(
        f"BigBird Pegasus socores:\nRouge-1 - {rouge_scores['eval_rouge-1']}\nRouge-2 - {rouge_scores['eval_rouge-2']}\nRouge-L - {rouge_scores['eval_rouge-L']}\n")
