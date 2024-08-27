from ..modules.moe_transformer_block import inject_prep_blocks


class PrepExpertModelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "reference"

    def apply(self, model):
        inject_prep_blocks(model.model.diffusion_model)
        return (model,)
