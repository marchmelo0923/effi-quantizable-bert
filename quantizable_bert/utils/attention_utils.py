__all__=['get_clipped_softmax_with_gamma','get_clipped_softmax_with_alpha']

import torch.nn as nn
from torch import clip
from functools import partial

from quantizable_bert.utils.message import assert_message

def clipped_softmax(attention_scores, zeta, gamma, dim=-1):
    # clip ((ζ − γ) · softmax(x) + γ, 0, 1)
    softmax_result = nn.functional.softmax(attention_scores, dim=dim)
    output = clip((zeta-gamma)*softmax_result+gamma, 0, 1)
    return output

def get_clipped_softmax_with_gamma(gamma, zeta, dim=-1):
    return partial(clipped_softmax, gamma=gamma, zeta=zeta, dim=dim)

def get_clipped_softmax_with_alpha(alpha, max_sequence_length, zeta, dim=-1):
    assert alpha>0, assert_message("attention_utils.py", "Value alpha should be larger than 0")
    gamma = alpha/max_sequence_length
    return get_clipped_softmax_with_gamma(gamma, zeta, dim)
