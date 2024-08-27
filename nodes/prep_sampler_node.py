import torch

import comfy.samplers
from comfy.samplers import KSAMPLER

from ..utils.sampler_utils import get_sampler_fn, create_prep_sampler


@torch.no_grad()
def sample_write(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    return model(x, sigmas[0] * s_in, **extra_args)


class PrepSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
            "expert": ([1,2,3,4],),
        }, "optional": {
        }}
    RETURN_TYPES = ("SAMPLER","ACT_BANK")
    FUNCTION = "build"

    CATEGORY = "segmoe"

    def build(self, sampler_name, expert):
        act_bank = {}
        sampler_fn = get_sampler_fn(sampler_name)
        sampler_fn = create_prep_sampler(sampler_fn, act_bank)
        sampler = KSAMPLER(sampler_fn)

        return (sampler, act_bank)
