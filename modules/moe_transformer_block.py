from typing import List
import torch
import torch.nn.functional as F
from torch import nn

from comfy.ldm.modules.attention import BasicTransformerBlock as OrigBasicTransformerBlock, optimized_attention
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

from ..utils.module_utils import isinstance_str

# mode = ['attn', 'ff', 'full']

class BasicTransformerBlock(OrigBasicTransformerBlock):
    def configure(self, block, idx, mode, expert_blocks, experts_per_token, weights=None):
        self.block = block
        self.idx = idx
        self.block_idx = (block, idx)
        self.mode = mode

        self.num_experts = len(expert_blocks)
        self.top_k = experts_per_token
        # self.out_dim = config.get("out_dim", self.hidden_dim)
        
        # gating
        hidden_dim = self.attn1.heads * self.attn1.dim_head
        self.gates =  nn.ModuleList([])
        if weights is not None:
            weights = weights[self.block_idx]
        if mode == 'attn' or mode == 'full':
            self.gate_attn1_hs_cond = nn.Linear(hidden_dim, self.num_experts, bias=False)
            self.gate_attn1_hs_cond.weight = nn.Parameter(weights[0]['attn1_hs'])
            self.gate_attn1_hs_uncond = nn.Linear(hidden_dim, self.num_experts, bias=False)
            self.gate_attn1_hs_uncond.weight = nn.Parameter(weights[1]['attn1_hs'])
            self.gate_attn2_hs_cond = nn.Linear(hidden_dim, self.num_experts, bias=False)
            self.gate_attn2_hs_cond.weight = nn.Parameter(weights[0]['attn2_hs'])
            self.gate_attn2_hs_uncond = nn.Linear(hidden_dim, self.num_experts, bias=False)
            self.gate_attn2_hs_uncond.weight = nn.Parameter(weights[1]['attn2_hs'])
            self.gate_attn2_ehs_cond = nn.Linear(hidden_dim, self.num_experts, bias=False)
            self.gate_attn2_ehs_cond.weight = nn.Parameter(weights[0]['attn2_ehs'])
            self.gate_attn2_ehs_uncond = nn.Linear(hidden_dim, self.num_experts, bias=False)
            self.gate_attn2_ehs_uncond.weight = nn.Parameter(weights[1]['attn2_ehs'])
            self.gates.extend([
                self.gate_attn1_hs_cond,
                self.gate_attn1_hs_uncond,
                self.gate_attn2_hs_cond,
                self.gate_attn2_hs_uncond,
                self.gate_attn2_ehs_cond,
                self.gate_attn2_ehs_uncond
            ])
        if mode == 'ff' or mode == 'full':
            self.gate_ff_cond = nn.Linear(hidden_dim, self.num_experts, bias=False)
            self.gate_ff_cond.weight = nn.Parameter(weights[0]['ff'])
            self.gate_ff_uncond = nn.Linear(hidden_dim, self.num_experts, bias=False)
            self.gate_ff_uncond.weight = nn.Parameter(weights[1]['ff'])
            self.gates.extend([
                self.gate_ff_cond,
                self.gate_ff_uncond
            ])

        for gate in self.gates:
            gate.requires_grad_(False)
        if expert_blocks:
            self.experts = nn.ModuleList(expert_blocks)
            self.experts.requires_grad_(False)

    def to(self, device):
        super().to(device)
        for gate in self.gates:
            gate.to(device)

        for expert in self.experts:
            expert.to(device)

        return self

    def _get_gate(self, cond, type):
        if type == 'attn1_hs':
            if cond == 0:
                return self.gate_attn1_hs_cond
            elif cond == 1:
                return self.gate_attn1_hs_uncond
        elif type == 'attn2_hs':
            if cond == 0:
                return self.gate_attn2_hs_cond
            elif cond == 1:
                return self.gate_attn2_hs_uncond
        elif type == 'attn2_ehs':
            if cond == 0:
                return self.gate_attn2_ehs_cond
            elif cond == 1:
                return self.gate_attn2_ehs_uncond
        elif type == 'ff':
            if cond == 0:
                return self.gate_ff_cond
            elif cond == 1:
                return self.gate_ff_uncond
        raise ValueError(f'No gate for cond {cond} and type {type}')
    
    def _get_routing_weights(self, x, cond, type):
        gate = self._get_gate(cond, type).to(x.device)
        batch_size, _, f_map_sz = x.shape
        x = x.view(-1, f_map_sz)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = gate(x)
        _, selected_experts = torch.topk(
            router_logits.sum(dim=0, keepdim=True), self.top_k, dim=1
        )
        routing_weights = F.softmax(
            router_logits[:, selected_experts[0]], dim=1, dtype=torch.float
        )

        return routing_weights.to(x.dtype), selected_experts
    
    def _mix_func(self, x_cond, routing_info, funcs):
        routing_weights, selected_experts = routing_info
        final_hidden_states = None
        batch_size, sequence_length, dim = x_cond.shape
        hidden_dim = funcs[0].weight.shape[0] if not hasattr(funcs[0], 'net') else funcs[0].net[2].weight.shape[0]
        final_hidden_states = torch.zeros(
            (batch_size , sequence_length, hidden_dim),
            dtype=x_cond.dtype,
            device=x_cond.device,
        )
        # Loop over all available experts in the model and perform the computation on each expert
        for i, expert_idx in enumerate(selected_experts[0].tolist()):
            current_hidden_states = routing_weights[:, i].view(
                batch_size, sequence_length, -1
            ) * funcs[expert_idx](x_cond)
            final_hidden_states += current_hidden_states

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states
    
    def _mix_attn1(self, x, transformer_options):
        if (self.mode != 'attn' and self.mode != 'full') or transformer_options.get('SAMPLE_TYPE', None) == 'PREP':
            return self.attn1.to_q(x), self.attn1.to_k(x), self.attn1.to_v(x)
        conds = transformer_options['cond_or_uncond']
        len_conds = len(conds)
        n_frames = len(x) // len_conds
        q_outputs = []
        k_outputs = []
        v_outputs = []
        for idx, cond in enumerate(conds):
            x_cond = x[idx*n_frames:(idx+1)*n_frames]
            routing_info = self._get_routing_weights(x_cond, cond, 'attn1_hs')
            q_cond = self._mix_func(x_cond, routing_info, [expert.attn1.to_q for expert in self.experts])
            q_outputs.append(q_cond)
            k_cond = self._mix_func(x_cond, routing_info, [expert.attn1.to_k for expert in self.experts])
            k_outputs.append(k_cond)
            v_cond = self._mix_func(x_cond, routing_info, [expert.attn1.to_v for expert in self.experts])
            v_outputs.append(v_cond)
        
        return torch.cat(q_outputs), torch.cat(k_outputs), torch.cat(v_outputs)
    
    def _mix_attn2(self, x, context, transformer_options):
        if (self.mode != 'attn' and self.mode != 'full') or transformer_options.get('SAMPLE_TYPE', None) == 'PREP':
            return self.attn2.to_q(x), self.attn2.to_k(context), self.attn2.to_v(context)
        conds = transformer_options['cond_or_uncond']
        len_conds = len(conds)
        n_frames = len(x) // len_conds
        q_outputs = []
        k_outputs = []
        v_outputs = []
        for idx, cond in enumerate(conds):
            x_cond = x[idx*n_frames:(idx+1)*n_frames]
            routing_info = self._get_routing_weights(x_cond, cond, 'attn2_hs')
            q_cond = self._mix_func(x_cond, routing_info, [expert.attn2.to_q for expert in self.experts])
            q_outputs.append(q_cond)
            context_cond = context[idx*n_frames:(idx+1)*n_frames]
            routing_info = self._get_routing_weights(context_cond, cond, 'attn2_ehs')
            k_cond = self._mix_func(context_cond, routing_info, [expert.attn2.to_k for expert in self.experts])
            k_outputs.append(k_cond)
            v_cond = self._mix_func(context_cond, routing_info, [expert.attn2.to_v for expert in self.experts])
            v_outputs.append(v_cond)
        
        return torch.cat(q_outputs), torch.cat(k_outputs), torch.cat(v_outputs)
    
    def _mix_ff(self, x, transformer_options):
        if (self.mode != 'ff' and self.mode != 'full') or transformer_options.get('SAMPLE_TYPE', None) == 'PREP':
            return self.ff(x)
        conds = transformer_options['cond_or_uncond']
        len_conds = len(conds)
        n_frames = len(x) // len_conds
        ff_outputs = []
        for idx, cond in enumerate(conds):
            x_cond = x[idx*n_frames:(idx+1)*n_frames]
            routing_info = self._get_routing_weights(x_cond, cond, 'ff')
            ff_cond = self._mix_func(x_cond, routing_info, [expert.ff for expert in self.experts])
            ff_outputs.append(ff_cond)
        
        return torch.cat(ff_outputs)
    
    def _attention_mechanism(self, q, k, v, heads):
        return optimized_attention(q, k, v, heads)

    def forward(self, x, context=None, transformer_options={}):
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head
        extra_options["attn_precision"] = self.attn_precision

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        n = self.norm1(x)

        conds = transformer_options['cond_or_uncond']
        len_conds = len(conds)
        n_frames = len(x) // len_conds

        sample_type = transformer_options.get('SAMPLE_TYPE', None)
        act_bank = transformer_options.get('ACT_BANK', None)

        if sample_type == 'PREP' and act_bank is not None:
            if self.block_idx not in act_bank:
                act_bank[self.block_idx] = { 
                    0: {
                        'attn1_hs': None,
                        'attn2_hs': None,
                        'attn2_ehs': None,
                        'ff': None
                    }, 1: {
                        'attn1_hs': None,
                        'attn2_hs': None,
                        'attn2_ehs': None,
                        'ff': None
                    }}
            for idx, cond in enumerate(conds):
                cond_n = n[idx*n_frames:(idx+1)*n_frames]
                act_bank[self.block_idx][cond]['attn1_hs'] = cond_n.clone().cpu()

        context_attn1 = n
        value_attn1 = context_attn1

        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            for p in patch:
                n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        if block_attn1 in attn1_replace_patch:
            q, context_attn1, value_attn1 = self._mix_attn1(n, transformer_options)
            hidden_states = attn1_replace_patch[block_attn1](q, context_attn1, value_attn1, extra_options)
            hidden_states = self.attn1.to_out(hidden_states)
            del q, context_attn1, value_attn1
        else:
            q, context_attn1, value_attn1 = self._mix_attn1(n, transformer_options)
            hidden_states = self.attn1.to_out(self._attention_mechanism(q,context_attn1, value_attn1, self.attn1.heads))
            del q, context_attn1, value_attn1
        
        n = hidden_states

        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n

        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)

        if self.attn2 is not None:
            n = self.norm2(x)
            if self.switch_temporal_ca_to_sa:
                context_attn2 = n
            else:
                context_attn2 = context

            if sample_type == 'PREP' and act_bank is not None:
                for idx, cond in enumerate(conds):
                    cond_n = n[idx*n_frames:(idx+1)*n_frames]
                    act_bank[self.block_idx][cond]['attn2_hs'] = cond_n.clone().cpu()
                    cond_context = context_attn2[idx*n_frames:(idx+1)*n_frames]
                    act_bank[self.block_idx][cond]['attn2_ehs'] = cond_context.clone().cpu()

            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for p in patch:
                    n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block

            if block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                n, context_attn2, value_attn2 = self._mix_attn2(n, context_attn2, transformer_options)
                n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                n = self.attn2.to_out(n)
                del context_attn2, value_attn2
            else:
                n, context_attn2, value_attn2 = self._mix_attn2(n, context_attn2, transformer_options)
                n = self.attn2.to_out(self._attention_mechanism(n,context_attn2, value_attn2, self.attn2.heads))
                del context_attn2, value_attn2

        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n
        if self.is_res:
            x_skip = x

        n = self.norm3(x)
        if sample_type == 'PREP' and act_bank is not None:
            for idx, cond in enumerate(conds):
                cond_n = n[idx*n_frames:(idx+1)*n_frames]
                act_bank[self.block_idx][cond]['ff'] = cond_n.clone().cpu()
        
        x = self._mix_ff(n, transformer_options)
        if self.is_res:
            x += x_skip
        return x


def _get_block_modules(module):
    blocks = list(filter(lambda x: isinstance_str(x[1], 'BasicTransformerBlock'), module.named_modules()))
    return [block for _, block in blocks]

def _clean_model(model):
    blocks = _get_block_modules(model)
    for block in blocks:
        if hasattr(block, 'experts'):
            block.experts = None


def inject_moe_blocks(models: List[UNetModel], mode, experts_per_token, gate_weights):
    for model in models:
        _clean_model(model)

    input_blocks = [_get_block_modules(model.input_blocks) for model in models]
    middle_blocks = [_get_block_modules(model.middle_block) for model in models]
    output_blocks = [_get_block_modules(model.output_blocks) for model in models]

    for i, block in enumerate(input_blocks[0]):
        block.__class__ = BasicTransformerBlock
        expert_blocks = [input_blocks[k][i] for k in range(len(models))]
        block.configure('input', i, mode, expert_blocks, experts_per_token, gate_weights)

    for i, block in enumerate(middle_blocks[0]):
        block.__class__ = BasicTransformerBlock
        expert_blocks = [middle_blocks[k][i] for k in range(len(models))]
        block.configure('middle', i, mode, expert_blocks, experts_per_token, gate_weights)

    for i, block in enumerate(output_blocks[0]):
        block.__class__ = BasicTransformerBlock
        expert_blocks = [output_blocks[k][i] for k in range(len(models))]
        block.configure('output', i, mode, expert_blocks, experts_per_token, gate_weights)

    return models[0]


def inject_prep_blocks(diffusion_model: UNetModel):
    input = _get_block_modules(diffusion_model.input_blocks)
    middle = _get_block_modules(diffusion_model.middle_block)
    output = _get_block_modules(diffusion_model.output_blocks)

    for i, block in enumerate(input):
        block.__class__ = BasicTransformerBlock
        block.configure('input', i, None, [], 0)

    for i, block in enumerate(middle):
        block.__class__ = BasicTransformerBlock
        block.configure('middle', i, None, [], 0)

    for i, block in enumerate(output):
        block.__class__ = BasicTransformerBlock
        block.configure('output', i, None, [], 0)
