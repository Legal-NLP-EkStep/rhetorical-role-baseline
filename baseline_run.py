import time
import gc
from datetime import datetime
from os import makedirs

import torch

from eval_run import eval_and_save_metrics
from utils import get_device, ResultWriter, log
from task import pubmed_task
from train import SentenceClassificationTrainer
from models import BertHSLN
import os

# BERT_VOCAB = "bert-base-uncased"
BERT_MODEL = "bert-base-uncased"
# BERT_VOCAB = "bert_model/scibert_scivocab_uncased/vocab.txt"
#BERT_MODEL = "allenai/scibert_scivocab_uncased"


config = {
    "bert_model": BERT_MODEL,
    "bert_trainable": False,
    "model": BertHSLN.__name__,
    "cacheable_tasks": [],

    "dropout": 0.5,
    "word_lstm_hs": 758,
    "att_pooling_dim_ctx": 200,
    "att_pooling_num_ctx": 15,

    "lr": 3e-05,
    "lr_epoch_decay": 0.9,
    "batch_size":  32,
    "max_seq_length": 128,
    "max_epochs": 20,
    "early_stopping": 5,

}

MAX_DOCS = -1
def create_task(create_func):
    return create_func(train_batch_size=config["batch_size"], max_docs=MAX_DOCS)

def create_generic_task(task_name):
    return generic_task(task_name, train_batch_size=config["batch_size"], max_docs=MAX_DOCS)

# ADAPT: Uncomment the task that has to be trained and comment all other tasks out

# ADAPT: Uncomment the task that has to be trained and comment all other tasks out
task = create_task(pubmed_task)
#task = create_task(pubmed_task_small)
#task = create_task(nicta_task)
#task = create_task(dri_task)
#task = create_task(art_task)
#task = create_task(art_task_small)
#task = create_generic_task(GEN_DRI_TASK)
#task = create_generic_task(GEN_PMD_TASK)
#task = create_generic_task(GEN_NIC_TASK)
# task = create_generic_task(GEN_ART_TASK)

# ADAPT: Set to False if you do not want to save the best model
save_best_models = True

# ADAPT: provide a different device if needed
device = get_device(0)

timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

# ADAPT: adapt the folder name of the run if necessary
run = f"{timestamp}_{task.task_name}_baseline"

# -------------------------------------------

os.makedirs("results/complete_epoch_wise_new",exist_ok=True)
#run_results = f'/nfs/data/sentence-classification/results/{run}'
run_results = f'results/{run}'
makedirs(run_results, exist_ok=False)

# preload data if not already done
task.get_folds()

restarts = 1 if task.num_folds == 1 else 1
for restart in range(restarts):
    for fold_num, fold in enumerate(task.get_folds()):
        start = time.time()
        result_writer = ResultWriter(f"{run_results}/{restart}_{fold_num}_results.jsonl")
        result_writer.log(f"Fold {fold_num} of {task.num_folds}")
        result_writer.log(f"Starting training {restart} for fold {fold_num}... ")

        trainer = SentenceClassificationTrainer(device, config, task, result_writer)
        best_model = trainer.run_training_for_fold(fold_num, fold, return_best_model=save_best_models)
        if best_model is not None:
            model_path = os.path.join(run_results, f'{restart}_{fold_num}_model.pt')
            result_writer.log(f"saving best model to {model_path}")
            torch.save(best_model.state_dict(), model_path)

        result_writer.log(f"finished training {restart} for fold {fold_num}: {time.time() - start}")

        # explicitly call garbage collector so that CUDA memory is released
        gc.collect()

log("Training finished.")

log("Calculating metrics...")
eval_and_save_metrics(run_results)
log("Calculating metrics finished")