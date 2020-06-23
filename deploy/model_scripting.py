import sys
sys.path.insert(0, '../')

import torch
from transformers import AlbertModel, AlbertConfig

from consonant.model.modeling import Consonant
from consonant.model.tokenization import NGRAMTokenizer


if __name__ == '__main__':

    ckpt = '../ckpt-0078000.bin'
    device = torch.device("cpu") #"cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device)
    print(state['ngram'])

    config = AlbertConfig(**state['config_dict'])
    config.attention_probs_dropout_prob = 0.0
    config.hidden_dropout_prob = 0.0
    print(config)
    model = Consonant(config)
    model.load_state_dict(state['model_state_dict'])

    tokenizer = NGRAMTokenizer(1)
    inputs = tokenizer.encode("sample text", max_char_length=100, return_attention_mask=True)
    input_ids = torch.tensor([inputs["head_ids"]], dtype=torch.long)


    traced_model = torch.jit.trace(model, [input_ids, input_ids])
    torch.jit.save(traced_model, "traced_model.pt")
    

