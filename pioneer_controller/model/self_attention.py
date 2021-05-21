import torch
import numpy as np
from torch.nn import functional as F

def self_attention(a, key_extractors, value_extractors, selector_extractors, attend_heads, hidden_dim):
    is_unsqueeze = False
    if len(a.shape) == 2:
        size = [1, a.shape[0], a.shape[1]]
        a = a.expand(*size)
        is_unsqueeze = True
    agents = range(len(a[0]))
    if len(a[0]) == 1:
        if is_unsqueeze == True:
            a = a.squeeze(0)
            return torch.zeros(a.shape).to('cuda')
        else:
            return a
    a = a.permute(1, 0, 2).contiguous()  # [batch_size, number_robots,dim] ->[number_robots,batch_size,dim]
    all_head_keys = [[k_ext(aa) for aa in a] for k_ext in key_extractors]
    all_head_values = [[k_ext(aa) for aa in a] for k_ext in value_extractors]
    all_head_selectors = [[k_ext(aa) for aa in a] for k_ext in selector_extractors]
    other_all_values = [[] for _ in range(len(agents))]
    all_attend_logits = [[] for _ in range(len(agents))]
    all_attend_probs = [[] for _ in range(len(agents))]

    for curr_head_keys, curr_head_values, curr_head_selectors in zip(
            all_head_keys, all_head_values,
            all_head_selectors):  # size of all_head_keys :[attend_heads,number_robots,dim]
        # iterate over agents
        for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
            keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
            values = [v for j, v in enumerate(curr_head_values) if j != a_i]
            # calculate attention across agents
            attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                         torch.stack(keys).permute(1, 2, 0))
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
            attend_weights = F.softmax(scaled_attend_logits, dim=2)
            other_values = (torch.stack(values).permute(1, 2, 0) *
                            attend_weights).sum(dim=2)
            other_all_values[i].append(other_values)
            all_attend_logits[i].append(attend_logits)
            all_attend_probs[i].append(attend_weights)
    # type(other_all_values):list of list of tensor:[num_robots,attend_heads,batch_size,attend_dim]
    other_all_values_To_Tensor = torch.stack(other_all_values[0]).to('cuda')
    for i in range(len(other_all_values)):
        if i > 0:
            other_all_values_To_Tensor = torch.cat((other_all_values_To_Tensor, torch.stack(other_all_values[i])))
    other_all_values_To_Tensor = other_all_values_To_Tensor.view(-1, attend_heads, other_all_values_To_Tensor.shape[1],
                                                                 other_all_values_To_Tensor.shape[2])

    other_all_values_To_Tensor = other_all_values_To_Tensor.permute(2, 0, 1, 3).contiguous()
    new_shape = other_all_values_To_Tensor.size()[:-2] + (hidden_dim,)
    b = other_all_values_To_Tensor.view(*new_shape)
    if is_unsqueeze == True:
        b = b.squeeze(0)
    return b

a = torch.rand(12,128)   #[num_robots,dim]
