import copy
import numpy as np
from collections import OrderedDict

class FL_scenario(object):
    def __init__(self, all_client_weights, n_clients, n_clients_each_round, distributed_dataloaders):
        self.all_client_weights = all_client_weights
        self.n_clients = n_clients
        self.n_clients_each_round = n_clients_each_round
        self.distributed_dataloaders = distributed_dataloaders
        np.random.seed(None)
        #self.personalized = personalized
        if self.n_clients != self.n_clients_each_round:
            self.type = 'cross_devices'
        else:
            self.type = 'cross_silo'
        #print(f"*****Current Federated Learning Scenario is {self.type}*****", file=output_file)

    def init_client_models(self, server_model, device='cuda'):
        models = [copy.deepcopy(server_model).to(device)
                  for _ in range(self.n_clients_each_round)]
        return models

    def init_personalized_model_weights(self, server_model, device='cuda'):
        personalized_model_weights = list()
        trainable = OrderedDict({key: None for key in server_model.trainable_keys})
        for key in server_model.trainable_keys:
            trainable[key] = copy.deepcopy(server_model.state_dict()[key].data)
        personalized_model_weights = [copy.deepcopy(trainable)
                                      for _ in range(self.n_clients)]
        return personalized_model_weights

    def cross_devices_random_selecting(self, client_sampling_type='uniform'):
        if client_sampling_type == 'poisson':
            sample_rate = self.n_clients_each_round / self.n_clients
            selected_client_index = np.where(np.random.rand(self.n_clients) < sample_rate)[0]
            if selected_client_index.size == 0:
                selected_client_index = np.array([np.random.randint(0, self.n_clients)])
            if selected_client_index.size > self.n_clients_each_round:
                selected_client_index = np.sort(
                    np.random.choice(selected_client_index, self.n_clients_each_round, replace=False)
                )
        else:
            selected_client_index = np.sort(np.random.choice(self.n_clients, self.n_clients_each_round, replace=False))
        selected_distributed_dataloaders = [self.distributed_dataloaders[i] for i in selected_client_index]
        selected_client_weights = self.all_client_weights[selected_client_index]
        return selected_client_index, selected_distributed_dataloaders, selected_client_weights