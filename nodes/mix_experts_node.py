from ..modules.moe_transformer_block import inject_moe_blocks
from ..utils.gate_utils import get_gate_weights


class MixExpertModelsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "mode": (["attn", "ff", "full"],),
            "experts_per_token": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            "model_base": ("MODEL",),
            "model2": ("MODEL",),
        }, 
        "optional": {
            "model3": ("MODEL",),
            "model4": ("MODEL",),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "reference"

    def apply(self, mode, experts_per_token, model_base, model2, model3=None, model4=None):
        models = [model_base, model2]
        if model3 is not None:
            models.append(model3)
        if model4 is not None:
            models.append(model4)

        act_banks = [model.model_options['transformer_options']['ACT_BANK'] for model in models]

        gate_weights = get_gate_weights(act_banks)
        diffusion_models = [model.model.diffusion_model for model in models]
        inject_moe_blocks(diffusion_models, mode, experts_per_token, gate_weights)
        return (model_base,)
