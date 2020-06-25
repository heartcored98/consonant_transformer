import sys
sys.path.insert(0, '../')

from abc import ABC
import json
import logging
import os


import torch
from transformers import AlbertModel, AlbertConfig
from ts.torch_handler.base_handler import BaseHandler

import consonant
from consonant.model.modeling import Consonant
from consonant.model.tokenization import NGRAMTokenizer

from tqdm import tqdm


logger = logging.getLogger(__name__)


class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """
    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        torch.set_num_threads(1)
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        setup_config_path = os.path.join(model_dir, "setup_config.json")

        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning('Missing the setup_config.json file.')

        #Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        #further setup config can be added.

        # model_pt_path = '../ckpt-0189000.bin'
        self.device = torch.device("cpu") #"cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path)
            self.tokenizer = NGRAMTokenizer(self.setup_config["ngram"])

        elif self.setup_config["save_mode"] == "pretrained":
            state = torch.load(model_pt_path, map_location=self.device)
            config = AlbertConfig(**state['config_dict'])
            self.model = Consonant(config)
            self.model.load_state_dict(state['model_state_dict'])
            self.tokenizer = NGRAMTokenizer(state["ngram"])

        else:
            logger.warning('Missing the checkpoint or state_dict.')
        

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_pt_path))
        self.initialized = True

    def preprocess(self, data):
        """ Basic text preprocessing, based on the user's chocie of application mode.
        """
        logger.info(str(data))
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")  # with txt file
            if isinstance(text, dict):
                logger.info(" ############## Got Dict !! ##########################")
                input_text = text['text']
            else:
                input_text = text.decode('utf-8')
        max_length = int(self.setup_config["max_length"])
        logger.info("Received text: '%s'", input_text)

        logger.info(input_text)
        # input_text = "안녕하세요? 반갑습니다. 오늘 날씨가 정말 끝내줘요. 너 너무 사랑스러워요"
        inputs = self.tokenizer.encode(input_text, max_char_length=max_length, return_attention_mask=True)
        return inputs

    def inference(self, inputs):
        """ Predict the class (or classes) of the received text using the serialized transformers checkpoint.
        """

        input_ids = torch.tensor([inputs["head_ids"]], dtype=torch.long).to(self.device)
        attention_masks = torch.tensor([inputs["attention_masks"]], dtype=torch.bool).to(self.device)
        
        # Handling inference for sequence_classification.
        with torch.no_grad():
            output = self.model(input_ids, attention_masks)
        predict_label = output[0].argmax(dim=2)
        predict_string = self.tokenizer.decode_sent(input_ids[0].detach().cpu().numpy(), predict_label[0].detach().cpu().numpy())

        logger.info("Model predicted: '%s'", predict_string)
        return [{'predict': predict_string}]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TransformersSeqClassifierHandler()


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


if __name__ == "__main__":
    _service.initialize(None)
    for i in tqdm(range(100)):
        data = _service.preprocess(None)
        data = _service.inference(data)
        print(data)
