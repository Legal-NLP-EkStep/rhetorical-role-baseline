from prettytable import PrettyTable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


import torch
import random
import json
import time
import models
import numpy as np
import os

from eval import eval_model
from utils import tensor_dict_to_gpu, tensor_dict_to_cpu, ResultWriter, get_num_model_parameters, print_model_parameters
from task import Task, Fold
import gc
import copy

class SentenceClassificationTrainer:
    '''Trainer for baseline model and also for Sequantial Transfer Learning. '''
    def __init__(self, device, config, task: Task, result_writer:ResultWriter):
        self.device = device
        self.config = config
        self.result_writer = result_writer
        self.cur_result = dict()
        self.cur_result["task"] = task.task_name
        self.cur_result["config"] = config

        self.labels = task.labels
        self.task = task

    def write_results(self, fold_num, epoch, train_duration, dev_metrics, dev_confusion, test_metrics, test_confusion):
        self.cur_result["fold"] = fold_num
        self.cur_result["epoch"] = epoch
        self.cur_result["train_duration"] = train_duration
        self.cur_result["dev_metrics"] = dev_metrics
        self.cur_result["dev_confusion"] = dev_confusion
        self.cur_result["test_metrics"] = test_metrics
        self.cur_result["test_confusion"] = test_confusion

        self.result_writer.write(json.dumps(self.cur_result))


    def run_training_for_fold(self, fold_num, fold: Fold, initial_model=None, return_best_model=False):

        self.result_writer.log(f'device: {self.device}')

        train_batches, dev_batches, test_batches = fold.train, fold.dev, fold.test

        self.result_writer.log(f"fold: {fold_num}")
        self.result_writer.log(f"train batches: {len(train_batches)}")
        self.result_writer.log(f"dev batches: {len(dev_batches)}")
        self.result_writer.log(f"test batches: {len(test_batches)}")

        # instantiate model per reflection
        if initial_model is None:
            model = getattr(models, self.config["model"])(self.config, [self.task])
        else:
            self.result_writer.log("Loading weights from initial model....")
            model = copy.deepcopy(initial_model)
            # for transfer learning do not transfer the output layer
            model.reinit_output_layer([self.task], self.config)

        self.result_writer.log("Model: " + model.__class__.__name__)
        self.cur_result["model"] = model.__class__.__name__

        model.to(self.device)

        max_train_epochs = self.config["max_epochs"]
        lr = self.config["lr"]
        max_grad_norm = 1.0

        self.result_writer.log(f"Number of model parameters: {get_num_model_parameters(model)}")
        self.result_writer.log(f"Number of model parameters bert: {get_num_model_parameters(model.bert)}")
        self.result_writer.log(f"Number of model parameters word_lstm: {get_num_model_parameters(model.word_lstm)}")
        self.result_writer.log(f"Number of model parameters attention_pooling: {get_num_model_parameters(model.attention_pooling)}")
        self.result_writer.log(f"Number of model parameters sentence_lstm: {get_num_model_parameters(model.sentence_lstm)}")
        self.result_writer.log(f"Number of model parameters crf: {get_num_model_parameters(model.crf)}")
        print_model_parameters(model)

        # for feature based training use Adam optimizer with lr decay after each epoch (see Jin et al. Paper)
        optimizer = Adam(model.parameters(), lr=lr)
        epoch_scheduler = StepLR(optimizer, step_size=1, gamma=self.config["lr_epoch_decay"])

        best_dev_result = 0.0
        early_stopping_counter = 0
        epoch = 0
        early_stopping = self.config["early_stopping"]
        best_model = None
        optimizer.zero_grad()
        while epoch < max_train_epochs and early_stopping_counter < early_stopping:
            epoch_start = time.time()

            self.result_writer.log(f'training model for fold {fold_num} in epoch {epoch} ...')

            random.shuffle(train_batches)
            # train model
            model.train()
            for batch_num, batch in enumerate(train_batches):
                # move tensor to gpu
                tensor_dict_to_gpu(batch, self.device)

                output = model(
                    batch=batch,
                    labels=batch["label_ids"]
                )
                loss = output["loss"]
                loss = loss.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # move batch to cpu again to save gpu memory
                tensor_dict_to_cpu(batch)

                if batch_num % 100 == 0:
                    self.result_writer.log(f"Loss in fold {fold_num}, epoch {epoch}, batch {batch_num}: {loss.item()}")

            train_duration = time.time() - epoch_start

            epoch_scheduler.step()

            # evaluate model
            results={}
            self.result_writer.log(f'evaluating model...')
            dev_metrics, dev_confusion,labels_dict, _ = eval_model(model, dev_batches, self.device, self.task)
            results['dev_metrics']=dev_metrics
            results['dev_confusion'] = dev_confusion
            results['labels_dict'] = labels_dict
            results['classification_report']=_


            if dev_metrics[self.task.dev_metric] > best_dev_result:
                if return_best_model:
                    best_model = copy.deepcopy(model)
                best_dev_result = dev_metrics[self.task.dev_metric]
                early_stopping_counter = 0
                self.result_writer.log(f"New best dev {self.task.dev_metric} {best_dev_result}!")
                results={}
                test_metrics, test_confusion,labels_dict,_ = eval_model(model, test_batches, self.device, self.task)
                results['dev_metrics']=dev_metrics
                results['dev_confusion'] = dev_confusion
                results['labels_dict'] = labels_dict
                results['classification_report']=_


                self.write_results(fold_num, epoch, train_duration, dev_metrics, dev_confusion, test_metrics, test_confusion)
                self.result_writer.log(
                    f'*** fold: {fold_num},  epoch: {epoch}, train duration: {train_duration}, dev {self.task.dev_metric}: {dev_metrics[self.task.dev_metric]}, test weighted-F1: {test_metrics["weighted-f1"]}, test macro-F1: {test_metrics["macro-f1"]}, test accuracy: {test_metrics["acc"]}')
            else:
                early_stopping_counter += 1
                self.result_writer.log(f'fold: {fold_num}, epoch: {epoch}, train duration: {train_duration}, dev {self.task.dev_metric}: {dev_metrics[self.task.dev_metric]}')

            epoch += 1
        return best_model



class SentenceClassificationMultitaskTrainer:
    '''Trainer for multitask model.
       Has only small differences to  SentenceClassificationTrainer
        (i.e. no early stopping, two devices to separate models on several gpus)
    '''
    def __init__(self, device, config, tasks, result_writer, device2=None):
        self.device = device
        self.device2 = device2
        self.config = config
        self.result_writer = result_writer
        self.cur_result = dict()
        self.cur_result["tasks"] = [task.task_name for task in tasks]
        self.cur_result["config"] = config

        self.tasks = tasks

    def write_results(self, task, epoch, train_duration, dev_metrics, dev_confusion, test_metrics, test_confusion):
        self.cur_result["task"] = task.task_name
        self.cur_result["epoch"] = epoch
        self.cur_result["train_duration"] = train_duration
        self.cur_result["dev_metrics"] = dev_metrics
        self.cur_result["dev_confusion"] = dev_confusion
        self.cur_result["test_metrics"] = test_metrics
        self.cur_result["test_confusion"] = test_confusion

        self.result_writer.write(json.dumps(self.cur_result))


    def run_training(self, train_batches, dev_batches, test_batches, restart, fold_num, save_models=False, save_best_model_path=None):

        self.result_writer.log(f'device: {self.device}')

        train_batch_count = len(train_batches)
        self.result_writer.log(f"train batches: {train_batch_count}")
        self.result_writer.log(f"dev batches: {len(dev_batches)}")
        self.result_writer.log(f"test batches: {len(test_batches)}")

        # instantiate model per reflection
        model = getattr(models, self.config["model"])(self.config, self.tasks)
        self.result_writer.log("Model: " + model.__class__.__name__)
        self.cur_result["model"] = model.__class__.__name__

        if self.device2 is not None:
            model.to_device(self.device, self.device2)
        else:
            model.to(self.device)

        max_train_epochs = self.config["max_epochs"]
        lr = self.config["lr"]
        max_grad_norm = 1.0

        # for feature based training use Adam optimizer with lr decay after each epoch (see Jin et al. Paper)
        optimizer = Adam(model.parameters(), lr=lr)
        epoch_scheduler = StepLR(optimizer, step_size=1, gamma=self.config["lr_epoch_decay"])

        optimizer.zero_grad()

        best_dev_result = 0.0
        epoch = 0
        while epoch < max_train_epochs:
            epoch_start = time.time()

            self.result_writer.log(f'training model in epoch {epoch} ...')

            random.shuffle(train_batches)
            # train model
            model.train()
            for batch_num, batch in enumerate(train_batches):
                # move tensor to gpu
                tensor_dict_to_gpu(batch, self.device)

                output = model(batch=batch, labels=batch["label_ids"])
                loss = output["loss"]
                loss = loss.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # move batch to cpu again to save gpu memory
                tensor_dict_to_cpu(batch)

                if batch_num % 100 == 0:
                    self.result_writer.log(f"Loss in epoch {epoch}, batch {batch_num}: {loss.item()}")

            train_duration = time.time() - epoch_start

            epoch_scheduler.step()

            # evaluate model
            weighted_f1_dev_scores = []
            for task in self.tasks:
                self.result_writer.log(f'evaluating model for task {task.task_name}...')
                dev_metrics, dev_confusion, _ = eval_model(model, dev_batches, self.device, task)
                test_metrics, test_confusion, _ = eval_model(model, test_batches, self.device, task)
                self.write_results(task, epoch, train_duration, dev_metrics, dev_confusion, test_metrics, test_confusion)
                self.result_writer.log(
                    f'epoch: {epoch}, train duration: {train_duration}, dev weighted f1: {dev_metrics["weighted-f1"]}, dev {task.dev_metric}: {dev_metrics[task.dev_metric]}, test weighted-F1: {test_metrics["weighted-f1"]}, test micro-F1: {test_metrics["micro-f1"]}. test macro-F1: {test_metrics["macro-f1"]}, test accuracy: {test_metrics["acc"]}')
                weighted_f1_dev_scores.append(test_metrics["weighted-f1"])
            weighted_f1_avg = np.mean(weighted_f1_dev_scores)

            if save_models:
                model_copy = copy.deepcopy(model)
                model_path = os.path.join(save_best_model_path, f'{restart}_{fold_num}_{epoch}_model.pt')
                self.result_writer.log(f"saving model to {model_path}")
                torch.save(model_copy.state_dict(), model_path)

            if weighted_f1_avg > best_dev_result:
                best_dev_result = weighted_f1_avg
                self.result_writer.log(f'*** epoch: {epoch}, mean weighted-F1 dev score: {weighted_f1_avg}')
            else:
                self.result_writer.log(f'epoch: {epoch}, mean weighted-F1 dev score: {weighted_f1_avg}')

            epoch += 1

