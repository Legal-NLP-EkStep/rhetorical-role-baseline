import math
import os
import json
import numpy as np
import torch
from tensorboardX import SummaryWriter

import distributed
from models.reporter_ext import ReportMgr, Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str
from rouge_score import rouge_scorer
from tqdm import tqdm
from prepro.data_builder import BertData

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, args, model, optim,
                 grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = torch.nn.BCELoss(reduction='none')
        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        # step =  self.optim._step + 1
        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                sentence_rhetorical_roles = batch.sentence_rhetorical_roles

                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls,sentence_rhetorical_roles)

                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """

        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()
        sent_score_labels = []
        file_chunk_sent_scores={} ## key is filename and value is list of sentences containing sentence scores
        can_path = '%s_step%d.candidate' % (self.args.result_path, step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        summaries_folder = self.args.result_path + '/predicted_summaries/'
        if not os.path.exists(summaries_folder):
            os.makedirs(summaries_folder)
        sent_scores_path = '%s_step%d_sent_scores.pt' % (self.args.result_path, step)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        src = batch.src
                        labels = batch.src_sent_labels
                        segs = batch.segs
                        clss = batch.clss
                        mask = batch.mask_src
                        mask_cls = batch.mask_cls
                        unique_ids = batch.unique_ids
                        sentence_rhetorical_roles = batch.sentence_rhetorical_roles

                        gold = []
                        pred = []
                        ids = []

                        if (cal_lead):
                            selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                        elif (cal_oracle):
                            selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                            range(batch.batch_size)]
                        else:
                            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls,sentence_rhetorical_roles)

                            loss = self.loss(sent_scores, labels.float())
                            loss = (loss * mask.float()).sum()
                            batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                            stats.update(batch_stats)

                            sent_scores = sent_scores.cpu().data.numpy()
                            file_name,chunk_id = batch.unique_ids[0].split('___')
                            chunk_id = int(chunk_id)
                            src_labels = list(labels.cpu().numpy()[0])
                            if type(sent_scores[0]) == np.float32:
                                sent_scores = np.array([sent_scores])
                            sent_scores_list = list(sent_scores[0])
                            sent_rhetorical_roles_list = list(sentence_rhetorical_roles.cpu().data.numpy()[0])
                            for sent_id,(sent_txt,sent_label,sent_score,sent_rhet_role) in enumerate(zip(batch.src_str[0],src_labels,sent_scores_list,sent_rhetorical_roles_list)):
                                if file_chunk_sent_scores.get(file_name) is None:
                                    file_chunk_sent_scores[file_name]=[]
                                sent_dict = {'file_name':file_name,'chunk_id':chunk_id,'sent_id':sent_id,'sent_txt':sent_txt,
                                             'sent_score':sent_score,'sent_label':sent_label,'sent_rhetorical_role':sent_rhet_role}
                                file_chunk_sent_scores[file_name].append(sent_dict)


        lawbriefs_summary_map = {1:"facts",2:"facts",3:"arguments",4:"arguments",
                                 5:"issue",6:"ANALYSIS",7:'ANALYSIS',
                                    8:'ANALYSIS',9:'ANALYSIS',10:'decision',11:'decision'}##### keys are baseline rhetorical roles and values are LawBriefs roles
        lawbriefs_categories =["PREAMBLE",'facts','issue','arguments','ratio','decision']
        predicted_categories = ['facts','issue','arguments','ANALYSIS','decision']
        additional_mandaroty_categories = ['issue','decision']

        def create_concatenated_summaries(file_chunk_sent_scores):
            #### this function accepts the sentence scores and returns predicted and gold summary texts
            predicted_summaries = []
            gold_summaries = []
            predicted_summaries_rr = []
            rr_summaries=[]
            for file_name,sent_list in file_chunk_sent_scores.items():

                if self.args.use_rhetorical_roles and self.args.seperate_summary_for_each_rr:
                    # ######## take top N sentences for each rhetorical role
                    file_rr_sents ={} ##### keys are rhetorical roles and values are dict of {'sentences':[],'token_cnt':100}
                    for sent_dict in sent_list:
                        sent_token_cnt = len(sent_dict['sent_txt'].split())
                        sent_rr= lawbriefs_summary_map[sent_dict['sent_rhetorical_role']]
                        if file_rr_sents.get(sent_rr) is None:
                            file_rr_sents[sent_rr] = {'sentences':[sent_dict],'token_cnt':sent_token_cnt}
                        else:
                            file_rr_sents[sent_rr]['sentences'].append(sent_dict)
                            file_rr_sents[sent_rr]['token_cnt']+=sent_token_cnt

                    min_token_cnt_per_rr = 50 ######## if original text for a rhetorical role is below this then it is not summarized.
                    selected_sent_list=[]
                    for rr,sentences_dict in file_rr_sents.items():
                        if sentences_dict['token_cnt']<=min_token_cnt_per_rr:
                            selected_sent_list.extend(sentences_dict['sentences'])
                        else:
                            rr_sorted_sent_list = sorted(sentences_dict['sentences'], key=lambda x: x['sent_score'], reverse=True)
                            sents_to_keep = math.ceil(self.args.summary_sent_precent * len(sentences_dict['sentences']) / 100)
                            rr_selected_sent = rr_sorted_sent_list[:sents_to_keep]
                            rr_selected_sent = sorted(rr_selected_sent, key=lambda x: (x['chunk_id'], x['sent_id']))
                            selected_sent_list.extend(rr_selected_sent)

                else:
                    ######### take top N sentences by combining all the rhetorical roles
                    sent_list = sorted(sent_list,key =lambda x: x['sent_score'],reverse=True)
                    sents_to_keep = math.ceil(self.args.summary_sent_precent * len(sent_list)/100)
                    selected_sent_list = sent_list[:sents_to_keep]
                    selected_sent_list = sorted(selected_sent_list,key = lambda x: (x['chunk_id'],x['sent_id']))


                predicted_summary='' ###### to be used  for rouge  calculation
                predicted_summary_rr = {} ## keys are rhetorical role and values are concatenated sentences
                ## create predicted summary
                rhetorical_roles_in_test = list(set([lawbriefs_summary_map[i['sent_rhetorical_role']] for i in selected_sent_list if i['sent_label']==1]))
                for sent_dict in selected_sent_list:
                    if self.args.rogue_exclude_roles_not_in_test and self.args.use_rhetorical_roles:
                        #########  if the selected sentence has rhetorical role not in test then exclude it
                        if lawbriefs_summary_map[sent_dict['sent_rhetorical_role']] in rhetorical_roles_in_test:
                            if predicted_summary=='':
                                predicted_summary = sent_dict['sent_txt']
                            else:
                                predicted_summary = predicted_summary + '\n' + sent_dict['sent_txt']
                    else:
                        if predicted_summary == '':
                            predicted_summary = sent_dict['sent_txt']
                        else:
                            predicted_summary = predicted_summary + '\n' + sent_dict['sent_txt']

                    sent_lawbriefs_role= lawbriefs_summary_map[sent_dict['sent_rhetorical_role']]
                    if predicted_summary_rr.get(sent_lawbriefs_role) is None:
                        predicted_summary_rr[sent_lawbriefs_role] = sent_dict['sent_txt']
                    else:
                        predicted_summary_rr[sent_lawbriefs_role] = predicted_summary_rr[sent_lawbriefs_role] + '\n' +sent_dict['sent_txt']

                ######## copy the additional mandatory roles to summary
                if self.args.use_rhetorical_roles and self.args.add_additional_mandatory_roles_to_summary and not self.args.seperate_summary_for_each_rr :
                    sent_list = sorted(sent_list,key = lambda x: (x['chunk_id'],x['sent_id']))
                    for category in additional_mandaroty_categories:
                        category_sentences = [i for i in sent_list if lawbriefs_summary_map[i['sent_rhetorical_role']]==category]
                        if category_sentences:
                            if predicted_summary_rr.get(category) is not None:
                                ###### remove the category as it may not have all the sentences.
                                predicted_summary_rr.pop(category)
                            for cat_sent in category_sentences:
                                if predicted_summary_rr.get(category) is None:
                                    predicted_summary_rr[category] = cat_sent['sent_txt']
                                else:
                                    predicted_summary_rr[category] = predicted_summary_rr[category] + '\n' + \
                                                                                cat_sent['sent_txt']



                predicted_summaries.append(predicted_summary)

                ## create gold summary
                gold_summary = ''
                gold_summary_rr = {}
                sent_list = sorted(sent_list, key=lambda x: (x['chunk_id'],x['sent_id']))
                for sent_dict in sent_list:
                    if sent_dict['sent_label']==1:
                        if gold_summary=='':
                            gold_summary = sent_dict['sent_txt']
                        else:
                            gold_summary = gold_summary + '\n' + sent_dict['sent_txt']

                        sent_lawbriefs_role = lawbriefs_summary_map[sent_dict['sent_rhetorical_role']]
                        if gold_summary_rr.get(sent_lawbriefs_role) is None:
                            gold_summary_rr[sent_lawbriefs_role] = sent_dict['sent_txt']
                        else:
                            gold_summary_rr[sent_lawbriefs_role] = gold_summary_rr[sent_lawbriefs_role] + '\n' + sent_dict['sent_txt']
                gold_summaries.append(gold_summary)

                rr_summaries.append({'predicted_summary':predicted_summary_rr , 'gold_summary':gold_summary_rr ,'file_name':file_name})
            return predicted_summaries,gold_summaries,rr_summaries

        torch.save(sent_score_labels, sent_scores_path)

        final_candidate,final_gold,rr_summaries = create_concatenated_summaries(file_chunk_sent_scores)

        print('********* Writing summaries to ', summaries_folder)

        ##### try to read original summaries

        original_summaries_path = os.path.split(self.args.bert_data_path)[0]+'/lawbriefs_summary_fulltext_map.json'

        if os.path.exists(original_summaries_path):
            original_summaries = json.load(open(original_summaries_path))
        else:
            original_summaries=[]

        with open(summaries_folder + 'predicted_summaries.json', 'w') as file:
            json.dump(rr_summaries,file)
        for pred_summary in rr_summaries:
            summary_txt=''
            if original_summaries:
                summary_txt = summary_txt + '\n-------------- PREAMBLE --------------------------------\n'
                for original_summary in original_summaries:
                    if original_summary['file_name'] == pred_summary['file_name']:
                        if original_summary.get('PREAMBLE') is not None:
                            summary_txt = summary_txt + original_summary['PREAMBLE'] + '\n\n'
                        break

            summary_txt =summary_txt+'\n-------------- Predicted summary --------------------------------\n'

            for predicted_category in predicted_categories:
                if pred_summary['predicted_summary'].get(predicted_category) is not None:
                    summary_txt = summary_txt + '\n#############'+ predicted_category +' summary ################\n'
                    summary_txt = summary_txt + pred_summary['predicted_summary'].get(predicted_category) + '\n\n'


            summary_txt =summary_txt + '\n-------------- LawBrief mapped summary --------------------------------\n'
            for rr_lawbrief,rr_text in pred_summary['gold_summary'].items():
                summary_txt = summary_txt + '\n#############'+ rr_lawbrief +' summary ################\n'
                summary_txt = summary_txt + rr_text + '\n\n'

            with open(summaries_folder+os.path.splitext(pred_summary['file_name'])[0]+'.txt', 'w') as file:
                file.write(summary_txt.encode("ascii", "ignore").decode())



        def rouge(label_sent: list, pred_sent: list) -> dict:
            """Approximate ROUGE scores, always run externally for final scores."""
            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeLsum"],
                use_stemmer=True)
            r1, r2, rl = 0.0, 0.0, 0.0
            for ls, ps in tqdm(zip(label_sent, pred_sent), desc=f"Calculating scores on {len(label_sent)} labels",
                               total=len(label_sent)):
                score = scorer.score(ls, ps)
                r1 += score["rouge1"].fmeasure
                r2 += score["rouge2"].fmeasure
                rl += score["rougeLsum"].fmeasure
            result = {"eval_rouge-1": r1 / len(label_sent),
                      "eval_rouge-2": r2 / len(label_sent),
                      "eval_rouge-L": rl / len(label_sent)}
            return result

        if (step != -1 and self.args.report_rouge):
            # rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            rouges = rouge(final_gold,final_candidate)
            for key in rouges.keys():
                logger.info(f'{key}:{rouges[key]}')
        self._report_step(0, step, valid_stats=stats)

        return stats

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            labels = batch.src_sent_labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask_src
            mask_cls = batch.mask_cls
            sentence_rhetorical_roles = batch.sentence_rhetorical_roles

            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls,sentence_rhetorical_roles)

            loss = self.loss(sent_scores, labels.float())
            loss = (loss * mask.float()).sum()
            (loss / loss.numel()).backward()
            # loss.div(float(normalization)).backward()

            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
