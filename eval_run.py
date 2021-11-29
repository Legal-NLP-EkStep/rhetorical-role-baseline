import glob
import json
import pandas as pd
import numpy as np
import os
from task import pubmed_task
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename, format="pdf")
    return ax


def load_best_results(run_path, task):
    '''Loads the relevant metrics for a certain task with the best dev set performance.'''
    results = []
    for best_metrics in load_best_dev_metrics(run_path, task):
        epoch = best_metrics["epoch"]

        results.append([
            best_metrics["dev_metrics"]["weighted-f1"],
            best_metrics["dev_metrics"]["acc"],
            best_metrics["dev_metrics"]["macro-f1"],
            best_metrics["test_metrics"]["weighted-f1"],
            best_metrics["test_metrics"]["acc"],
            best_metrics["test_metrics"]["macro-f1"]])

    return results


def load_best_dev_metrics(run_path, task_name):
    '''Loads the results with the best dev set performance for the given task.'''
    dev_metric = get_task(task_name).dev_metric
    results = []
    for fn in sorted(list(glob.glob(f'{run_path}/*.jsonl'))):
        with open(fn, "r") as f:
            # print(fn)
            best_dev = 0
            best_metrics = None
            epoch = 0
            m1 = re.match("(\\d*)_(\\d*)_.*", os.path.basename(fn))
            m2 = re.match("(\\d*)_.*", os.path.basename(fn))
            if m1:
                restart = int(m1.group(1))
                fold_num = int(m1.group(2))
            elif m2:
                restart = int(m2.group(1))
                fold_num = int(m2.group(1))
            else:
                continue

            for line in f:
                metrics = json.loads(line)
                if metrics["task"] != task_name:
                    continue

                if metrics["dev_metrics"][dev_metric] > best_dev or best_metrics is None:
                    best_dev = metrics["dev_metrics"][dev_metric]
                    best_metrics = metrics
                epoch = metrics["epoch"]

            best_metrics["fold_num"] = fold_num
            best_metrics["restart"] = restart
            results.append(best_metrics)
    return results

def create_generic_task(task_name):
    return generic_task(task_name, train_batch_size=1, max_docs=-1)

def get_all_tasks():
    result = []
    result.append(pubmed_task(train_batch_size=-1, max_docs=-1))
    result.append(pubmed_task_small(train_batch_size=-1, max_docs=-1))
    result.append(nicta_task(train_batch_size=-1, max_docs=-1))
    result.append(dri_task(train_batch_size=-1, max_docs=-1))
    result.append(art_task(train_batch_size=-1, max_docs=-1))
    result.append(art_task_small(train_batch_size=-1, max_docs=-1))

    result.append(create_generic_task(GEN_DRI_TASK))
    result.append(create_generic_task(GEN_PMD_TASK))
    result.append(create_generic_task(GEN_NIC_TASK))
    result.append(create_generic_task(GEN_ART_TASK))

    return result


def get_task(taskname):
    for t in get_all_tasks():
        if t.task_name == taskname:
            return t
    return None


def load_tasks_in_run(run_path):
    tasks = set()
    for fn in glob.glob(f'{run_path}/*.jsonl'):
        with open(fn, "r") as f:
            for line in f:
                metrics = json.loads(line)
                tasks.add(metrics["task"])
    return list(sorted(tasks))


def calc_f1(cm):
    # True Positives are on the diagonal position
    true_pos = np.diag(cm)
    # False positives are column-wise sums. Without the diagonal
    false_pos = np.sum(cm, axis=0) - true_pos
    # False negatives are row-wise sums. Without the diagonal.
    false_neg = np.sum(cm, axis=1) - true_pos

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def load_confusion_matrix(run_path, task_name, absolute=True):
    cms = []
    for best_dev_results in load_best_dev_metrics(run_path, task_name):

        if absolute:
            cm = best_dev_results["test_metrics"]["confusion_abs"]
            cm = np.array(cm)
        else:
            cm = best_dev_results["test_confusion"]
            cm = np.array(cm)

        # remove mask class
        cm = cm[1:, 1:]
        cms.append(cm)
    # average the cm across folds resp. restarts
    cm_avg = np.round(np.mean(cms, axis=0)).astype(int)
    return cm_avg


def eval_and_save_metrics(path):
    '''
    Calculates averaged metrics across folds resp. restarts for all tasks in the run and saves
    them in results.csv and f1_per_label.csv.
    Besides a confusion matrix is calculated for each task and saved as "{task_name}_cm.pdf".
    :param path: Path to the run. In this path also the files are saved,
    '''
    tasks_in_run = load_tasks_in_run(path)
    task_metrics = []
    for task in tasks_in_run:
        task_results = load_best_results(path, task)
        means = np.round(np.mean(np.array(task_results) * 100, axis=0), 2)
        stds = np.round(np.std(np.array(task_results) * 100, axis=0), 3)
        r = [task + " mean"] + means.tolist()
        task_metrics.append(r)
        r = [task + " std"] + stds.tolist()
        task_metrics.append(r)

    metrics_columns = ["dev weighted-f1", "dev accuracy", "dev macro-f1", "test weighted-f1", "test accuracy", "test macro-f1"]
    result_df = pd.DataFrame(task_metrics, columns=["task"] + metrics_columns)
    result_df.to_csv(os.path.join(path, "results.csv"))

    f1_per_class = []
    for t in tasks_in_run:
        task_cm_abs = load_confusion_matrix(path, t, absolute=True)
        task_cm_rel = load_confusion_matrix(path, t, absolute=False)
        plot_confusion_matrix(
            task_cm_rel,
            get_task(t).labels[1:],
            title=t,
            normalize=True,
            filename=os.path.join(path, f'{t}_cm.pdf'))
        f1s = calc_f1(task_cm_abs)
        for i, f1 in enumerate(f1s):
            label_name = get_task(t).labels[1:][i]
            label_order = get_task(t).labels_pres.index(label_name)
            f1_per_class.append([t, label_order, label_name.title(), f1])

    f1_per_label_df = pd.DataFrame(f1_per_class, columns=["task", "order", "label", "F1"])
    f1_per_label_df = f1_per_label_df.sort_values(by=["task", "order"])
    f1_per_label_df.to_csv(os.path.join(path, "f1_per_label.csv"))

    return result_df, f1_per_label_df

