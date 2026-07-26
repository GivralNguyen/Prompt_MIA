import torch
import copy
import numpy as np
from Algo.aggregation import Agg
from Algo.nonparametric_aggregation import NonparametricAgg
import torch.nn as nn

def communication(server_model, models, client_weights, fed_method='fedavg', total_n_clietns=100, nonpara_hidden=128, device='cuda'):
    client_num = len(models)
    sum_weights = torch.tensor(np.sum(client_weights), dtype=torch.float, device=device)

    if fed_method == 'nonparametric_aggregation':
        for key in server_model.trainable_keys:
            if 'prompt' not in key and 'key' not in key:
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                temp = torch.div(temp, sum_weights)
                with torch.no_grad():
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(temp)
            elif 'prompt' in key and 'key' not in key:
                temp = list()
                prompt_dim = server_model.state_dict()[key].shape[-1]
                key_name = key.replace('prompt', 'key') if key.replace('prompt', 'key') in server_model.state_dict() else None
                temp_key = []
                if hasattr(server_model, 'trained_prompts_checklist'):
                    union_prompts_checklist = torch.zeros(server_model.state_dict()[key].shape[0], dtype=torch.int)
                    nonzero_index = torch.nonzero(server_model.trained_prompts_checklist).flatten()
                    union_prompts_checklist[nonzero_index] = 1
                    for client_idx in range(client_num):
                        nonzero_index = torch.nonzero(models[client_idx].trained_prompts_checklist).flatten()
                        union_prompts_checklist[nonzero_index] = 1
                    for client_idx in range(client_num):
                        chosen = torch.nonzero(union_prompts_checklist).flatten()
                        temp.append(copy.deepcopy(models[client_idx].state_dict()[key][chosen]))
                        if key_name is not None:
                            temp_key.append(copy.deepcopy(models[client_idx].state_dict()[key_name][chosen]))
                else:
                    for client_idx in range(client_num):
                        temp.append(copy.deepcopy(models[client_idx].state_dict()[key].squeeze()))
                        if key_name is not None:
                            temp_key.append(copy.deepcopy(models[client_idx].state_dict()[key_name].squeeze()))
                temp = torch.stack(temp, dim=0)
                agg = NonparametricAgg(prompt_dim, n_hidden=nonpara_hidden).to(device)
                temp, assignment, selected_idx = agg(temp, return_assignment=True)

                new_key = None
                if key_name is not None and len(temp_key) > 0:
                    temp_key = torch.stack(temp_key, dim=0)
                    key_slots = []
                    for centroid_idx in selected_idx:
                        mask = assignment[:, :, centroid_idx]
                        selected_keys = temp_key[mask.bool()]
                        if selected_keys.numel() == 0:
                            flat_idx = int(centroid_idx.item())
                            owner_client = flat_idx // temp_key.shape[1]
                            owner_slot = flat_idx % temp_key.shape[1]
                            key_slots.append(temp_key[owner_client, owner_slot])
                        else:
                            key_slots.append(selected_keys.mean(dim=0))
                    new_key = torch.stack(key_slots, dim=0)

                del agg
                with torch.no_grad():
                    if hasattr(server_model, 'trained_prompts_checklist'):
                        server_model.pool.prompt = nn.Parameter(temp, requires_grad=True)
                        for client_idx in range(client_num):
                            models[client_idx].pool.prompt = nn.Parameter(temp, requires_grad=True)
                        if new_key is not None:
                            server_model.pool.key = nn.Parameter(new_key, requires_grad=True)
                            for client_idx in range(client_num):
                                models[client_idx].pool.key = nn.Parameter(new_key, requires_grad=True)
                    else:
                        server_model.prompt_embeddings = nn.Parameter(temp, requires_grad=True)
                        for client_idx in range(client_num):
                            models[client_idx].prompt_embeddings = nn.Parameter(temp, requires_grad=True)
            else:
                continue
    elif fed_method == 'aggregation':
        for key in server_model.trainable_keys:
            if 'prompt' not in key and 'key' not in key:
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                temp = torch.div(temp, sum_weights)
                with torch.no_grad():
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(temp)
            elif 'prompt' in key and 'key' not in key:

                if key.replace('prompt', 'key') in server_model.state_dict():
                    key_name = key.replace('prompt', 'key')
                else:
                    raise KeyError(f"Cannot find key tensor for {key}")
        
                temp_prompt, temp_key = [], []
                # prompt_dim = server_model.state_dict()[key].shape[-1]
                if hasattr(server_model, 'trained_prompts_checklist'):
                    # union_prompts_checklist = torch.zeros(server_model.state_dict()[key].shape[0],dtype=torch.int)
                    # nonzero_index = torch.nonzero(server_model.trained_prompts_checklist).flatten()
                    # union_prompts_checklist[nonzero_index] = 1
                    # for client_idx in range(client_num):
                    #     nonzero_index = torch.nonzero(models[client_idx].trained_prompts_checklist).flatten()
                    #     union_prompts_checklist[nonzero_index] = 1
                    # for client_idx in range(client_num):
                    #     temp_prompt.append(copy.deepcopy(models[client_idx].state_dict()[key][torch.nonzero(union_prompts_checklist).flatten()]))
                    #     temp_key.append(copy.deepcopy(models[client_idx].state_dict()[key_name][torch.nonzero(union_prompts_checklist).flatten()]))
                    for client_idx in range(client_num):
                        temp_prompt.append(copy.deepcopy(models[client_idx].state_dict()[key].squeeze()))
                        temp_key.append(copy.deepcopy(models[client_idx].state_dict()[key_name].squeeze()))
                else:
                    for client_idx in range(client_num):
                        temp_prompt.append(copy.deepcopy(models[client_idx].state_dict()[key].squeeze()))
                        temp_key.append(copy.deepcopy(models[client_idx].state_dict()[key_name].squeeze()))

                temp_prompt = torch.stack(temp_prompt, dim=0)
                temp_key = torch.stack(temp_key, dim=0)
                # torch.save({
                #     "prompt": temp_prompt,
                #     "key": temp_key,
                # }, "aggregated_prompts.pth")
                
                agg = Agg().to(device)

                new_prompt = agg(temp_prompt)
                new_key = agg(temp_key)
                del agg
                torch.cuda.empty_cache()

                with torch.no_grad():
                    if hasattr(server_model, 'trained_prompts_checklist'):
                        print("\nnew_prompt.shape", new_prompt.shape)
                        server_model.pool.prompt = nn.Parameter(new_prompt, requires_grad=True)
                        for client_idx in range(client_num):
                            models[client_idx].pool.prompt = nn.Parameter(new_prompt, requires_grad=True)
                        print("new_key.shape", new_key.shape)
                        server_model.pool.key = nn.Parameter(new_key, requires_grad=True)
                        for client_idx in range(client_num):
                            models[client_idx].pool.key = nn.Parameter(new_key, requires_grad=True)
                    else:
                        server_model.prompt_embeddings = nn.Parameter(new_prompt, requires_grad=True)
                        for client_idx in range(client_num):
                            models[client_idx].prompt_embeddings = nn.Parameter(new_prompt, requires_grad=True)
    else:
        for key in server_model.trainable_keys:   #server_model.state_dict().keys():
            '''if 'num_batches_tracked' in key:
                server_model.state_dict()[key].dataa.copy_(models[0].state_dict()[key])
            else:'''
            temp = torch.zeros_like(server_model.state_dict()[key])
            for client_idx in range(client_num):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
            temp = torch.div(temp, sum_weights)
            with torch.no_grad():
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models