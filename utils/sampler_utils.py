import torch

import comfy.k_diffusion.sampling as k_diffusion_sampling


def create_prep_sampler(sample_fn, act_bank):
    @torch.no_grad()
    def sample(model, latents, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
        model_options = extra_args.get('model_options', {})
        transformer_options = model_options.get('transformer_options', {})

        model_options = {
            **model_options,
            'transformer_options': {
                **transformer_options,
                'SAMPLE_TYPE': 'PREP',
                'ACT_BANK': act_bank,
                'TOTAL_STEPS': 1,
            }
        }
        extra_args = {**extra_args, 'model_options': model_options}

        output = sample_fn(model, latents, sigmas, extra_args=extra_args, callback=callback, disable=disable, **extra_options)

        if 'REF_BANK' in model_options['transformer_options']:
            del model_options['transformer_options']['REF_BANK']

        if 'REF_TYPE' in model_options['transformer_options']:
            del model_options['transformer_options']['REF_TYPE']

        if 'TOTAL_STEPS' in model_options['transformer_options']:
            del model_options['transformer_options']['TOTAL_STEPS']

        return output
    
    return sample


def get_sampler_fn(sampler_name):
    if sampler_name == "dpm_fast":
        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            total_steps = len(sigmas) - 1
            return k_diffusion_sampling.sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_adaptive_function
    else:
        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
    return sampler_function
