"""Tokenizes the sentences with BertTokenizer as tokenisation costs some time.
"""

from transformers import BertTokenizer
from infer_new import write_in_hsln_format
BERT_VOCAB = "bert-base-uncased"
MAX_SEQ_LENGTH = 128


def tokenize_file(in_file, out_file, tokenizer):
    with open(in_file, encoding="utf-8") as in_f:
        with open(out_file, encoding="utf-8", mode="w") as out_f:
            for line in in_f:
                line = line.replace("\r", "")
                if line.strip() == "" or line.startswith("###"):
                    out_f.write(line + "\n")
                else:
                    ls = line.split("\t")
                    tag, sentence = ls[0], ls[1]
                    tokenized = tokenizer.encode(sentence, add_special_tokens=True, max_length=MAX_SEQ_LENGTH)
                    out_f.write(f'{tag}\t{" ".join([str(t) for t in tokenized])}\n')


def tokenize():
    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

    print("pubmed-20k")
    tokenize_file("datasets/pubmed-20k/train_clean.txt", "datasets/pubmed-20k/train_scibert.txt", tokenizer)
    tokenize_file("datasets/pubmed-20k/dev_clean.txt", "datasets/pubmed-20k/dev_scibert.txt", tokenizer)
    tokenize_file("datasets/pubmed-20k/test_clean.txt", "datasets/pubmed-20k/test_scibert.txt", tokenizer)


    for t in [ "PMD"]:
        print(f"{t}_generic")
        for s in ["train", "dev", "test"]:
            tokenize_file(f"datasets/{t}_generic/{s}_clean.txt", f"datasets/{t}_generic/{s}_scibert.txt", tokenizer)



tokenize()
