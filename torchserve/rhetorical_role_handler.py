import json
import logging
import os
import re
import uuid

import torch
from transformers import BertTokenizer
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException

import models
from data_prep import get_spacy_nlp_pipeline_for_indian_legal_text, seperate_and_clean_preamble, \
    get_judgment_text_pipeline, get_spacy_nlp_pipeline_for_preamble
from eval import eval_model
from infer_data_prep import split_into_sentences_tokenize_write
from infer_new import write_in_hsln_format
from models import BertHSLN
from task import pubmed_task

logger = logging.getLogger(__name__)


class RhetoricalRolePredictorHandler(BaseHandler):

    def __init__(self):
        super(RhetoricalRolePredictorHandler, self).__init__()
        self.initialized = False
        BERT_MODEL = "bert-base-uncased"
        self.config = {
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
            "batch_size": 32,
            "max_seq_length": 128,
            "max_epochs": 40,
            "early_stopping": 5,

        }

    def initialize(self, ctx):
        """ Loads the model.pt file and initialized the model object.
        Instantiates Tokenizer for preprocessor to use
        Loads labels to name mapping file for post-processing inference response
        """
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        self.model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        # Read model serialize/pt file
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(self.model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt or pytorch_model.bin file")

        # Make necessary directories
        hsln_format_txt_dirpath = 'datasets/pubmed-20k'
        os.makedirs(os.path.join(self.model_dir, hsln_format_txt_dirpath), exist_ok=True)
        self.hsln_format_txt_dirpath = os.path.join(self.model_dir, hsln_format_txt_dirpath)

        # Load model
        def create_task(create_func):
            return create_func(train_batch_size=self.config["batch_size"], max_docs=-1, data_folder=self.model_dir)

        task = create_task(pubmed_task)
        self.model = getattr(models, self.config["model"])(self.config, [task])
        self.model.load_state_dict(torch.load(model_pt_path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        logger.debug('Transformer model from path {0} loaded successfully'.format(self.model_dir))

        # Ensure to use the same tokenizer used during training
        BERT_VOCAB = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

        # set up nlp pipeline
        self.nlp = get_spacy_nlp_pipeline_for_indian_legal_text(model_name="en_core_web_trf",
                                                                disable=["attribute_ruler", "lemmatizer", 'ner'])

        self.nlp_judgment = get_judgment_text_pipeline()

        preamble_entities_list = ['COURT', 'PETITIONER', 'RESPONDENT', 'LAWYER', 'JUDGE']
        for preamble_entity in preamble_entities_list:
            self.nlp_judgment.vocab.strings.add(preamble_entity)

        ###### Pipeline for preamble
        self.nlp_preamble = get_spacy_nlp_pipeline_for_preamble(self.nlp_judgment.vocab)
        self.nlp_preamble.add_pipe("extract_preamble_entities", after="lemmatizer")
        del self.nlp_judgment
        import gc
        gc.collect()
        self.initialized = True


    def preprocess(self, data):
        """ Preprocessing input request by tokenizing
            Extend with your own preprocessing steps as needed
        """

        # convert incoming data into label studio format

        id = None
        text = ""
        token = None
        recieved_data = data[0].get("data")
        if recieved_data is None:
            recieved_data = data[0].get("body")

        if text is not None:
            text = recieved_data.get('text')
            id = recieved_data.get('id')
        if id is None:
            uid = uuid.uuid4()
            id = "RhetoricalRoleInference_" + str(uid.hex)

        try:
            sentences = text.decode('utf-8')
        except:
            sentences = text
        if type(sentences) is not str or not sentences:
            raise PredictionException("Missing text in input for processing", 516)

        # clean judgement
        preamble_text, preamble_end = seperate_and_clean_preamble(sentences, self.nlp_preamble)
        judgement_text = sentences[preamble_end:]
        judgement_text = re.sub(r'([^.\"\?])\n+ *', r'\1 ', judgement_text)
        # logger.info("Received text: '%s'", sentences)
        input_ls_format = [
            {'id': id, 'data': {'preamble_text': preamble_text, 'judgement_text': judgement_text,
                                'text': preamble_text + " " + judgement_text}}]

        # Tokenize the texts and write files
        split_into_sentences_tokenize_write(input_ls_format, os.path.join(self.model_dir, 'input_to_hsln.json'),
                                            self.nlp,
                                            hsln_format_txt_dirpath=self.hsln_format_txt_dirpath)

        write_in_hsln_format(os.path.join(self.model_dir, 'input_to_hsln.json'),
                             self.hsln_format_txt_dirpath, self.tokenizer)

        task = pubmed_task(train_batch_size=self.config["batch_size"], max_docs=-1,
                           data_folder=os.path.join(self.model_dir, 'datasets'))
        return task

    def inference(self, task):
        """ Predict the class of a text using a trained transformer model.
        """
        folds = task.get_folds()
        test_batches = folds[0].test
        metrics, confusion, labels_dict, class_report = eval_model(self.model, test_batches, self.device, task)
        filename_sent_boundries = json.load(
            open(os.path.join(self.hsln_format_txt_dirpath, 'sentece_boundries.json')))

        for doc_name, predicted_labels in zip(labels_dict['doc_names'], labels_dict['docwise_y_predicted']):
            filename_sent_boundries[doc_name]['pred_labels'] = predicted_labels

        with open(os.path.join(self.model_dir, 'input_to_hsln.json'), 'r') as f:
            input = json.load(f)
        for file in input:
            id = str(file['id'])
            pred_id = labels_dict['doc_names'].index(id)
            pred_labels = labels_dict['docwise_y_predicted']
            annotations = file['annotations']
            for i, label in enumerate(annotations[0]['result']):
                label['value']['labels'] = [pred_labels[pred_id][i]]

        return [input]

    def postprocess(self, inference_output):
        return inference_output


_service = RhetoricalRolePredictorHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
