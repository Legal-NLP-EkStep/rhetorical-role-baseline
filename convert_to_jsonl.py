from task import pubmed_task, nicta_task, NICTA_LABELS, PUBMED_LABELS, dri_task, art_task, NICTA_TASK, DRI_TASK, \
    ART_TASK, pubmed_task_small, art_task_small, GEN_DRI_TASK, GEN_PMD_TASK, GEN_NIC_TASK, GEN_ART_TASK, generic_task
import json



def write_docs(path, fold, settype, docs):
    with open(f"{path}/folds/{settype}_{fold}.jsonl", mode="w", encoding="utf-8") as f:
        for d in docs:
            jdoc = {
                "abstract_id": 0,
                "sentences": d.sentences,
                "labels": d.labels
            }
            f.write(f'{json.dumps(jdoc)}\n')

def write_folds_for_task(task):
    for fold, (train, dev, test) in enumerate(task.get_folds_examples(file_suffix='clean')):
        write_docs(task.data_dir, fold, "train", train)
        write_docs(task.data_dir, fold, "dev", dev)
        write_docs(task.data_dir, fold, "test", test)

#write_folds_for_task(dri_task(train_batch_size=32, max_docs=-1))
#write_folds_for_task(art_task(train_batch_size=32, max_docs=-1))
#write_folds_for_task(nicta_task(train_batch_size=32, max_docs=-1))
write_folds_for_task(pubmed_task(train_batch_size=32, max_docs=-1))




