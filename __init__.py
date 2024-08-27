from .nodes.mix_experts_node import MixExpertModelsNode
from .nodes.prep_end_node import PrepEndModelNode
from .nodes.prep_expert_model_node import PrepExpertModelNode
from .nodes.prep_sampler_node import PrepSamplerNode


NODE_CLASS_MAPPINGS = {
    "MixExpertModels": MixExpertModelsNode,
    "PrepEndModel": PrepEndModelNode,
    "PrepExpertModel": PrepExpertModelNode,
    "PrepSampler": PrepSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MixExpertModels": "MoE] Mix Experts",
    "PrepEndModel": "MoE] End Prep Model",
    "PrepExpertModel": "MoE] Start Prep Model",
    "PrepSampler": "MoE] Prep Sampler"
}
