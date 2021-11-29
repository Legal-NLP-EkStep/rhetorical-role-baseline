from os import makedirs

from sklearn.model_selection import train_test_split

from dataset_reader import InputDocument
from task import art_task, dri_task, pubmed_task, nicta_task


def map_labels(examples, label_map):
    result = []
    for i,d in enumerate(examples):
        print(i)
        try:
            mapped_labels = [label_map[l] for l in d.labels]
        except:
            import pdb;pdb.set_trace()
            print(d.sentences)
        mapped_doc = InputDocument(d.sentences, mapped_labels,d.doc_name)
        result.append(mapped_doc)
    return result

def create_train_dev_test_split(all_examples):
    # create split 70/10/20 for train/dev/test
    train_dev, test = train_test_split(all_examples, test_size=2/10, random_state=1)
    train, dev = train_test_split(train_dev, test_size=1/8, random_state=1)

    assert len(train) + len(dev) + len(test) == len(all_examples)
    return train, dev, test

def write_file(examples, out):
    print(f'writing file {out}...')
    with open(out, "w", encoding="utf-8") as f_out:
        for doc in examples:
            for s, l in zip(doc.sentences, doc.labels):
                f_out.write(f"{l}\t{s}\n")
            f_out.write("\n")

def truncate_examples(examples, portion):
    new_len = int(len(examples) * portion)
    print(
        f"Truncating training examp"
        f"les with factor {portion} from {len(examples)} to {new_len}")
    return examples[0: new_len]

def create_generic_dataset(task, label_map, truncate_portion):
    print(f'create generic dataset for {task.short_name} with portion {truncate_portion}')
    all_examples = task.get_all_examples(file_suffix="clean")
    import pdb;pdb.set_trace()

    all_examples = truncate_examples(all_examples, truncate_portion)
    import pdb;
    pdb.set_trace()

    mapped_examples = map_labels(all_examples, label_map)
    train, dev, test = create_train_dev_test_split(mapped_examples)
    import pdb;
    pdb.set_trace()
    print(f'train/dev/test for {task.short_name}: {len(train)}/{len(dev)}/{len(test)}')
    path = f"datasets/{task.short_name}_generic"
    makedirs(path, exist_ok=True)
    write_file(train, f"{path}/train_clean.txt")
    write_file(dev, f"{path}/dev_clean.txt")
    write_file(test, f"{path}/test_clean.txt")



PUBMED_LABELS_TO_GENERIC = {"PREAMBLE":"PREAMBLE","NONE":"NONE", "FAC":"FAC", "ISSUE":"ISSUE", "ARG_RESPONDENT": "ARG_RESPONDENT","ARG_PETITIONER":"ARG_PETITIONER", "ANALYSIS":"ANALYSIS", "PRE_RELIED":"PRE_RELIED","PRE_NOT_RELIED":"PRE_NOT_RELIED", "STA":"STA","RLC":"RLC","RPC":"RPC","RATIO":"RATIO"}
#PUBMED_LABELS_TO_GENERIC = {"mask": "mask", "BACKGROUND": "Background", "OBJECTIVE": "Problem", "METHODS":  "Contribution", "RESULTS": "Result", "CONCLUSIONS": "Conclusion"}
pmd = pubmed_task(1, -1)
create_generic_dataset(pmd, PUBMED_LABELS_TO_GENERIC, truncate_portion=1)

