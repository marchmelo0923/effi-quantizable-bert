import math
from typing import Optional, Tuple, Literal
from functools import partial

import torch
import torch.nn as nn


from quantizable_bert.utils.attention_utils import get_clipped_softmax_with_alpha, get_clipped_softmax_with_gamma

from transformers.models.bert.modeling_bert import BertForMaskedLM

from transformers import (
    AutoModelForMaskedLM,
)
from pathlib import Path

from quantizable_bert.quantizable_attention import MyBertSelfAttention

#########################################
# Referred from outlier-free-transformers
#########################################
def _replaceBertSelfAttention(config, model:BertForMaskedLM, softmax_fn, use_mlp_gating:bool, model_name_or_path:Optional[str]) -> BertForMaskedLM:
    """Replace bert model's self attention module with ours.

    Args:
        config (): Bert Config
        model (BertForMaskedLM): Original Bert model
        softmax_fn (function): defined softmax function (either partial or not)
        use_mlp_gating (bool): Use Gated Attention

    Returns:
        BertForMaskedLM: Replaced Bert
    """
    for layer_idx in range(len(model.bert.encoder.layer)):
        old_self = model.bert.encoder.layer[layer_idx].attention.self
        new_self = MyBertSelfAttention(
            config,
            softmax_fn=softmax_fn,
            use_mlp_gating=use_mlp_gating
        )
        if model_name_or_path is not None:
            new_self.load_state_dict(old_self.state_dict(), strict=False)
        model.bert.encoder.layer[layer_idx].attention.self = new_self
    return model

# Create your model from the config
def create_QuantizableBert(config,
                           model_cache_dir,
                           softmax_type:Literal['vanilla','clipped_alpha','clipped_gamma']='vanilla',
                           alpha:Optional[float]=None,
                           max_seq_length:Optional[float]=None,
                           gamma:Optional[float]=None,
                           zeta:Optional[float]=None,
                           use_mlp_gating:bool=False,
                           model_name_or_path:Optional[str]=None
                           )->BertForMaskedLM:
    """Create new bert model with clipped_softmax and gated_attention.

    Args:
        config (_type_): _description_
        model_cache_dir: huggingface's model caching directory
        softmax_type (Literal[&#39;vanilla&#39;,&#39;clipped_alpha&#39;,&#39;clipped_gamma&#39;], optional): Set 'vanilla', if you don't use Clipped Softmax. Otherwise, set to 'clipped_alpha' or 'clipped_gamma' for your preferences.
        alpha (Optional[float]): Value for softmax_type='clipped_alpha'. Please check Quantization-Transformer paper's Section 5.2
        max_seq_length (Optional[float]): Value for softmax_type='clipped_alpha'. Value for softmax_type='clipped_alpha'. Please check Quantization-Transformer paper's Section 5.2
        gamma (Optional[float]): Value for softmax_type='clipped_gamma'. Value for softmax_type='clipped_alpha'. Please check Quantization-Transformer paper's Section 5.1
        zeta (Optional[float]): Value for softmax_type='clipped_gamma' and 'clipped_alpha' both.
        use_mlp_gating (bool): Use Gated Attention
        model_name_or_path (Optional[str]): using checkpoint path

    Returns:
        BertForMaskedLM: _description_
    """
    if model_name_or_path:
        origin_bert = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=model_cache_dir,
        )
    else:
        origin_bert = AutoModelForMaskedLM.from_config(config)

    assert softmax_type in {'vanilla', 'clipped_alpha', 'clipped_gamma'}, 'softmax_type should be one of {"vanilla", "clipped_alpha", "clipped_gamma"}'
    _softmax_fn = None
    # set softmax function
    if softmax_type=='vanilla':
        _softmax_fn = nn.functional.softmax
    elif softmax_type=='clipped_alpha':
        assert alpha is not None and max_seq_length is not None and zeta is not None, 'clipped_alpha require alpha, max_seq_length, zeta args'
        _softmax_fn = get_clipped_softmax_with_alpha(alpha=alpha, max_sequence_length=max_seq_length, zeta=zeta)
    elif softmax_type=='clipped_gamma':
        assert gamma is not None and zeta is not None, 'clipped_alpha require alpha, max_seq_length, zeta args'
        _softmax_fn = get_clipped_softmax_with_gamma(gamma=gamma, zeta=zeta)
    
    my_bert = _replaceBertSelfAttention(config=config,
                                        model=origin_bert,
                                        softmax_fn=_softmax_fn,
                                        use_mlp_gating=use_mlp_gating,
                                        model_name_or_path=model_name_or_path
    )

    # Gating -> load the model again to load missing alpha
    if model_name_or_path is not None and use_mlp_gating:
        state_dict = torch.load(str(Path(model_name_or_path) / "pytorch_model.bin"))
        new_state_dict = {}
        for name, val in state_dict.items():
            if "gate_list" in name:
                new_state_dict[name] = val
        my_bert.load_state_dict(new_state_dict, strict=False)

    return my_bert