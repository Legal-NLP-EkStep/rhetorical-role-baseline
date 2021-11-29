import json
import glob


def read_docs_from_jsonl(fn):
    result = []
    with open(fn, "r", encoding="utf-8") as f_in:
        for l in f_in:
            d = json.loads(l)
            result.append(d)
    return result

def convert_dataset_to_csv(dataset):
    docs = []
    docs.extend(read_docs_from_jsonl(f"datasets/{dataset}/folds/train_0.jsonl"))
    docs.extend(read_docs_from_jsonl(f"datasets/{dataset}/folds/dev_0.jsonl"))
    docs.extend(read_docs_from_jsonl(f"datasets/{dataset}/folds/test_0.jsonl"))
    with open(f"datasets/{dataset}/prev_gold_labels.csv", "w", encoding="utf-8") as f_out:
        f_out.write("prev_label,label\n")
        for d in docs:
            for i in range(len(d["labels"])):
                if i == 0:
                    prev_label = "START"
                else:
                    prev_label = d["labels"][i-1]
                cur_label = d["labels"][i]
                f_out.write(f"{prev_label},{cur_label}\n")
    print("finished: " + dataset)


convert_dataset_to_csv("ART")
convert_dataset_to_csv("DRI")



