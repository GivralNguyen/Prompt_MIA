import torch
from torch import nn
import numpy as np
from Algo.communication import communication
from util.train_eval import train, evaluate, evaluate_mask
from util.print_info import print_epoch_end
# from util.TOdevice import to_device

class fedavg(nn.Module):
    def __init__(self, server_model, scenario, loss_fun, class_mask, fed_method='fedavg', nonpara_hidden=128, device='cuda'):
        super(fedavg, self).__init__()
        self.server_model = server_model
        self.server_model.eval()
        self.scenario = scenario
        self.client_model = self.scenario.init_client_models(server_model, device=device)
        self.loss_fun = loss_fun
        self.fed_method = fed_method
        self.nonpara_hidden = nonpara_hidden
        self.class_mask = class_mask
        self.device = device
        self.history = [list() for _ in range(scenario.n_clients)]
        self.selected_client_index = np.arange(self.scenario.n_clients_each_round)
        self.selected_distributed_dataloaders = self.scenario.distributed_dataloaders[:self.scenario.n_clients_each_round]
        self.selected_client_weights = self.scenario.all_client_weights[:self.scenario.n_clients_each_round]

    '''def fit_in_one_epoch(self, idx, epochs, train_opt_func, optimizer, print_output):
        for epoch in range(epochs):
            self.client_model[idx].train()'''

    def client_train(self, comm_round, epochs, lr, dp_mechanism, eps, frac,
                     opt_func=torch.optim.Adam, reduce_sim_scalar=0.01,
                     print_output=False, defense='none', mixup_alpha=0.2,
                     sparse_ratio=0.0, perturb_sigma=0.0,
                     sampling_proportion=1.0, client_sampling_type='uniform'):
        if self.scenario.type == 'cross_devices':
            self.selected_client_index, self.selected_distributed_dataloaders, self.selected_client_weights \
            = self.scenario.cross_devices_random_selecting(client_sampling_type=client_sampling_type)
        sampled_clients = len(self.selected_client_index)
        for i in range(sampled_clients):
            torch.cuda.empty_cache()
            optimizer = opt_func(filter(lambda p : p.requires_grad, self.client_model[i].parameters()), lr,
                                 betas=(0.9, 0.98), eps=1e-6)
            # if print_output:
            #     print(f'------------Client_{self.selected_client_index[i]+1} start local trainig------------', 
            #           file=output_file)
            if self.class_mask is not None:
                mask = self.class_mask[(self.selected_client_index[i]//20)]
                # print(mask, file=output_file)

            for epoch in range(epochs):
                self.client_model[i].train()
                if self.class_mask is not None:
                    l, t, a = train(self.client_model[i], self.selected_distributed_dataloaders[i],  # noqa: E741
                                optimizer, dp_mechanism, eps, frac, epochs, self.loss_fun,
                                self.device, reduce_sim_scalar, mask,
                                defense=defense, mixup_alpha=mixup_alpha,
                                sparse_ratio=sparse_ratio, perturb_sigma=perturb_sigma,
                                sampling_proportion=sampling_proportion)
                else:
                    l, t, a = train(self.client_model[i], self.selected_distributed_dataloaders[i],  # noqa: E741
                                    optimizer, dp_mechanism, eps, frac, epochs, self.loss_fun,
                                    self.device, reduce_sim_scalar,
                                    defense=defense, mixup_alpha=mixup_alpha,
                                    sparse_ratio=sparse_ratio, perturb_sigma=perturb_sigma,
                                    sampling_proportion=sampling_proportion)
                # if print_output:
                #     print_epoch_end(epoch, l, t, a, output_file)
            
            # Add noise to prompts after local update 
            if dp_mechanism == 'Prompt':
                self.client_model[i].pool.add_noise_to_prompt(epsilon = 5.0)
                                        
            if hasattr(self.client_model[i], 'trained_prompts_checklist'):
                self.client_model[i].trained_prompts_checklist /= torch.max(self.client_model[i].trained_prompts_checklist)
                # print(torch.nonzero(self.client_model[i].trained_prompts_checklist).flatten(), file=output_file)
                # print(self.client_model[i].trained_prompts_checklist, file=output_file)

        # print(f"Trained Prompts Checklist of Client {i}: {self.client_model[i].trained_prompts_checklist}")
        # print(f"Trained Prompts Checklist of Server: {self.server_model.trained_prompts_checklist}")

    def server_aggre(self):
        sampled_clients = len(self.selected_client_index)
        active_client_models = self.client_model[:sampled_clients]
        self.server_model, active_client_models = communication(self.server_model, active_client_models,
                                                                self.selected_client_weights, self.fed_method,
                                                                nonpara_hidden=self.nonpara_hidden,
                                                                device=self.device)
        self.client_model[:sampled_clients] = active_client_models
        if hasattr(self.server_model, 'trained_prompts_checklist'):
            self.server_model.reset_trained_pormpts_checklist()
            sampled_clients = len(self.selected_client_index)
            for i in range(sampled_clients):
                self.client_model[i].reset_trained_pormpts_checklist()
                
        prompt_pool_size = self.server_model.pool.prompt.shape
        print(f'Server Prompt Pool Size: {prompt_pool_size}')

    # def adversarial_inject(self, dataloader, n_sampled_clients, n_tokens, query_model_type):
    #     x, label = next(iter(dataloader))
    #     x = to_device(x, self.device).float()
    #     label = to_device(label, self.device).long()

    #     with torch.no_grad():
    #         cls_features = self.server_model.get_cls_keys(x)
    #         if query_model_type == 'resnet50':
    #             adj_prompt = self.server_model.convnet_proj(cls_features)
    #         elif query_model_type == 'efficientnet_b0':
    #             adj_prompt = self.server_model.efficientnet_b0_proj(cls_features)
    #         elif query_model_type == 'bit':
    #             adj_prompt = self.server_model.bit_proj(cls_features)
            
    #     query_vector = cls_features.repeat(n_tokens, 1)
    #     prompt_vector = adj_prompt.repeat(n_tokens, 1)
        
    #     for i in range(n_sampled_clients):
    #         with torch.no_grad():
    #             real_prompt = self.server_model.pool.prompt
    #             print("Before Inject Prompt Length:", real_prompt.shape)
    #             updated_prompt = torch.cat([real_prompt, prompt_vector], dim=0)
    #             self.client_model[i].pool.prompt = nn.Parameter(updated_prompt, requires_grad=True)
    #             print("After Inject Prompt Length:", updated_prompt.shape)

    #             real_key = self.server_model.pool.key
    #             print("Before Inject Key Length:", real_key.shape)
    #             updated_key = torch.cat([real_key, query_vector], dim=0)
    #             self.client_model[i].pool.key = nn.Parameter(updated_key, requires_grad=True)
    #             print("After Inject Key Length:", updated_key.shape)

    #             old_checklist = self.client_model[i].trained_prompts_checklist
    #             old_len = old_checklist.shape[0]
    #             new_len = updated_prompt.size(0)
    #             new_checklist = torch.zeros(new_len, dtype=torch.float32, device=self.device)
    #             new_checklist[:old_len] = old_checklist
    #             self.client_model[i].trained_prompts_checklist = new_checklist

    #     self.server_model.pool.prompt = nn.Parameter(updated_prompt, requires_grad=True)
    #     old_checklist = self.server_model.trained_prompts_checklist
    #     old_len = old_checklist.shape[0]
    #     new_len = updated_prompt.size(0)
    #     new_checklist = torch.zeros(new_len, dtype=torch.float32, device=self.device)
    #     new_checklist[:old_len] = old_checklist
    #     self.server_model.trained_prompts_checklist = new_checklist

    def client_eval(self, testloader):
        # client_eval_list = list()
        sampled_clients = len(self.selected_client_index)
        for i in range(sampled_clients):
            train_loss, train_acc = evaluate(self.client_model[i], testloader, self.loss_fun, self.device)
            # client_eval_list.append([train_loss, train_acc])
            print(f'Client_{self.selected_client_index[i]+1}: Train_loss: {train_loss}; Accuracy: {train_acc}')
        # return client_eval_list

    def server_eval(self, testloader, nround):
        if self.class_mask is not None:
     
            eval_loss, eval_acc = evaluate_mask(self.server_model, testloader, self.loss_fun, self.device, self.class_mask)
        else:
            eval_loss, eval_acc = evaluate(self.server_model, testloader, self.loss_fun, self.device)
        print(f"Comm_round_{nround+1} Server model: Eval_loss: {eval_loss}; Accuracy: {eval_acc}")
        # print(f'Comm_round_{nround+1} Server model: Eval_loss: {eval_loss}; Accuracy: {eval_acc}')
        if hasattr(self.server_model, 'trained_prompts_checklist'):
            # print(torch.nonzero(self.server_model.trained_prompts_checklist).flatten())
            # print(self.server_model.trained_prompts_checklist)
            return eval_loss, eval_acc, len(self.server_model.trained_prompts_checklist)
        else:
            return eval_loss, eval_acc
        
    def server_eval_(self, testloader):
        if self.class_mask is not None:
            # print(len(testloader), file=output_file)
            eval_loss, eval_acc = evaluate_mask(self.server_model, testloader, self.loss_fun, self.device, self.class_mask)
        else:
            eval_loss, eval_acc = evaluate(self.server_model, testloader, self.loss_fun, self.device)
        print(f"Server model: Eval_loss: {eval_loss}; Accuracy: {eval_acc}")
        # print(f'Server model: Eval_loss: {eval_loss}; Accuracy: {eval_acc}', file=output_file)
        if hasattr(self.server_model, 'trained_prompts_checklist'):
            # print(torch.nonzero(self.server_model.trained_prompts_checklist).flatten(), file=output_file)
            # print(self.server_model.trained_prompts_checklist, file=output_file)
            return eval_loss, eval_acc, len(self.server_model.trained_prompts_checklist)
        else:
            return eval_loss, eval_acc