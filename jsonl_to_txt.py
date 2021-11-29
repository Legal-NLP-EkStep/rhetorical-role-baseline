import json
import glob

def convert_jsonl_to_txt(fn):
    with open(fn, "r", encoding="utf-8") as f_in:
        tfn = fn.replace(".jsonl", ".txt")
        with open (tfn, "w", encoding="utf-8") as f_out:
            for l in f_in:
                d = json.loads(l)
                for sentence, label in zip(d["sentences"], d["labels"]):
                    f_out.write(f'{label}\t{sentence}\n')
                f_out.write('\n')

for fn in glob.glob("datasets/**/folds/*.jsonl"):
    print(fn)
    convert_jsonl_to_txt(fn)