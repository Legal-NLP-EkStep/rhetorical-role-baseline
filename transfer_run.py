import time
import gc
from datetime import datetime
from os import makedirs

from eval_run import eval_and_save_metrics
from utils import get_device, log, ResultWriter
from task import pubmed_task, nicta_task, NICTA_LABELS, PUBMED_LABELS, dri_task, art_task, NICTA_TASK, DRI_TASK, \
    ART_TASK
import os


from train import SentenceClassificationTrainer
from models import BertHSLN
import torch


# BERT_VOCAB = "bert-base-uncased"
# BERT_MODEL = "bert-base-uncased"
# BERT_VOCAB = "bert_model/scibert_scivocab_uncased/vocab.txt"
BERT_MODEL = "allenai/scibert_scivocab_uncased"

config = {
    "bert_model": BERT_MODEL,
    "bert_trainable": False,
    "model": BertHSLN.__name__,
    "cacheable_tasks": [NICTA_TASK, DRI_TASK, ART_TASK],

    "dropout": 0.5,
    "word_lstm_hs": 758,
    "att_pooling_dim_ctx": 200,
    "att_pooling_num_ctx": 15,

    "lr": 3e-05,
    "lr_epoch_decay": 0.9,
    "batch_size": 32,
    "max_seq_length": 128,
    "max_epochs": 20,
    "early_stopping": 5,

    # ADAPT: do not transfer context enriching  (INIT 2), otherwise INIT 1
    "without_context_enriching_transfer": True
}

MAX_DOCS = -1
def create_task(create_func):
    return create_func(train_batch_size=config["batch_size"], max_docs=MAX_DOCS)


# ADAPT: Uncomment the primary task (source task) and the secondary task (target task) and comment all other tasks out
#primary_task = create_task(pubmed_task)
# primary_task = create_task(nicta_task)
primary_task = create_task(dri_task)
# primary_task = create_task(art_task)

#secondary_task = create_task(dri_task)
secondary_task = create_task(nicta_task)
# secondary_task = create_task(art_task)
# secondary_task = create_task(pubmed_task)

# ADAPT: provide the path of the baseline run for the resp primary task
primary_task_run_folder = "results/2019-12-10_10_28_14_DRI_baseline"

# ADAPT: Set to False if you do not want to save the best model
save_best_models = True

timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

# ADAPT: adapt the folder name of the run if necessary
run = f"{timestamp}_transfer_{primary_task.task_name}_to_{secondary_task.task_name}"

# ADAPT: provide a different device if needed
device = get_device(0)
# -------------------------------------------

run_results = f'results/{run}'
makedirs(run_results, exist_ok=False)


def train_secondary_task(initial_model, restart, fold_num, fold):
    start = time.time()
    result_writer = ResultWriter(f"{run_results}/{restart}_{fold_num}_{secondary_task.task_name}_results.jsonl")
    result_writer.log(f"Training {restart} in fold {fold_num} of secondary task {secondary_task.task_name}")

    trainer = SentenceClassificationTrainer(device, config, secondary_task, result_writer)
    best_model = trainer.run_training_for_fold(fold_num, fold, initial_model=initial_model, return_best_model=save_best_models)
    if best_model is not None:
        model_path = os.path.join(run_results, f'{restart}_{fold_num}_model.pt')
        result_writer.log(f"saving best model to {model_path}")
        torch.save(best_model.state_dict(), model_path)

    result_writer.log(f"finished training {restart} in fold {fold_num}: {time.time() - start}")

    # explicitly call garbage collector so that CUDA memory is released
    gc.collect()


log(f'num fold in task {secondary_task.task_name}: {secondary_task.num_folds}')

for train_num in range(max(3, secondary_task.num_folds)):
    if primary_task.num_folds == 1:
        prim_restart = train_num % 3
        prim_fold = 0
    else:
        prim_restart = 0
        prim_fold = train_num % primary_task.num_folds

    initial_model_path = os.path.join(primary_task_run_folder, f'{prim_restart}_{prim_fold}_model.pt')

    log(f'loading model {initial_model_path}')
    initial_model = BertHSLN(config, [primary_task])
    initial_model.load_state_dict(torch.load(initial_model_path, map_location=device))

    if secondary_task.num_folds == 1:
        train_fold_num = 0
        train_restart = train_num % 3
    else:
        train_fold_num = train_num
        train_restart = 0

    train_fold = secondary_task.get_folds()[train_fold_num]
    train_secondary_task(initial_model, train_restart, train_fold_num, train_fold)



log("Training finished.")

log("Calculating metrics...")
eval_and_save_metrics(run_results)
log("Calculating metrics finished")


