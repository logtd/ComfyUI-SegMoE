import torch


def get_gate_weights(act_banks):
    gate_vects = {}
    block_idxs = list(act_banks[0].keys())
    conds = list(act_banks[0][block_idxs[0]].keys())
    names = list(act_banks[0][block_idxs[0]][conds[0]].keys())
    for block_idx in block_idxs:
        gate_vects[block_idx] = {}
        for cond in conds:
            gate_vects[block_idx][cond] = {}
            for name in names:
                states = []
                for bank in act_banks:
                    hs = bank[block_idx][cond][name][0]
                    hs = hs.sum(dim=0) / hs.shape[0]
                    hs /= hs.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                    states.append(hs)
                gate_vects[block_idx][cond][name] = torch.stack(states)
    return gate_vects
