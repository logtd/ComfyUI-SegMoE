import random


class PrepEndModelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "latent": ("LATENT",),
            "act_bank": ("ACT_BANK",),
        }, 
        "optional": {
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "reference"

    def apply(self, model, latent, act_bank):
        model = model.clone()

        rand_int = random.randint(0,10)

        transformer_options = {**model.model_options.get('transformer_options', {})}
        model.model_options = { **model.model_options, rand_int: act_bank }

        transformer_options['ACT_BANK'] = act_bank

        model.model_options['transformer_options'] = transformer_options

        return (model,)