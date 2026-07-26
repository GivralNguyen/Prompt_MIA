import os
import argparse
import random
import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
import torchvision
from tqdm._tqdm import trange
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
import pytz
from datetime import datetime
from Algo.get_algos import get_algorithm
from data_util.DataTransformBuilder import build_data_transform
from data_util.DataDistributer import DataPartitioner
from data_util.TinyImageNet_reader import TinyImageNet_reader
from data_util.PACS_reader import Pacs_reader
from data_util.Four_dataset_reader import four_dataset_reader
from Scenario import FL_scenario
from util.TOdevice import to_device
from Models.L2P_heuristic_model import L2P_ViT_B32, L2P_CaiT_S24, L2P_DeiT_B16, L2P_ConViT_Base
from util.saving_tools import save_eval_npy_file, save_model_trainable_part,load_model_trainable_part
# from util.dp_mech import LaplaceMechanism, GaussianMechanism,AnalyticGaussianMechanism, StaircaseMechanism

import torch
import copy
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from Algo.nonparametric_aggregation import NonparametricAgg
# from Algo.normal_aggregation import NormalAgg
import torch.nn as nn

def broadcast_like_communication(server_model, client_models, device='cuda'):
    """
    Copy server_model params into each client in the same style as `communication(...)`:
    - For normal params: use `.data.copy_(...)`
    - For prompt/key tensors: replace with new `nn.Parameter(...)` like in your code
    """
    client_num = len(client_models)

    for key in server_model.trainable_keys:
    
        if 'prompt' not in key and 'key' not in key:
            temp = server_model.state_dict()[key]  # already a tensor on correct device
            with torch.no_grad():
                for client_idx in range(client_num):
                    client_models[client_idx].state_dict()[key].data.copy_(temp)
                    
        elif 'prompt' in key and 'key' not in key:
            # your code checks the paired key tensor exists in state_dict
            if key.replace('prompt', 'key') in server_model.state_dict():
                key_name = key.replace('prompt', 'key')
            else:
                raise KeyError(f"Cannot find key tensor for {key}")

            # If model exposes pool.{prompt,key} (has trained_prompts_checklist)
            if hasattr(server_model, 'trained_prompts_checklist'):
                with torch.no_grad():
                    server_prompt = server_model.pool.prompt.detach()
                    server_key    = server_model.pool.key.detach()
                    # update server to keep consistency with the pattern
                    server_model.pool.prompt = nn.Parameter(server_prompt, requires_grad=True)
                    server_model.pool.key    = nn.Parameter(server_key,    requires_grad=True)
                    for client_idx in range(client_num):
                        client_models[client_idx].pool.prompt = nn.Parameter(server_prompt.clone(), requires_grad=True)
                        client_models[client_idx].pool.key    = nn.Parameter(server_key.clone(),    requires_grad=True)
            else:
                # fallback path in your code: prompt_embeddings only
                with torch.no_grad():
                    server_prompt = server_model.prompt_embeddings.detach()
                    server_model.prompt_embeddings = nn.Parameter(server_prompt, requires_grad=True)
                    for client_idx in range(client_num):
                        client_models[client_idx].prompt_embeddings = nn.Parameter(server_prompt.clone(), requires_grad=True)
        else:
            # if you ever list raw 'key' tensors in trainable_keys without 'prompt' in name
            # just push them like normal params
            temp = server_model.state_dict()[key]
            with torch.no_grad():
                for client_idx in range(client_num):
                    client_models[client_idx].state_dict()[key].data.copy_(temp)
                    
if __name__ == '__main__':
    # Federated learning settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default="0",
                    help='Comma-separated GPU device IDs, e.g., "0", "0,1", or "" for CPU')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Algorithm to choose: [fedavg ]')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset to choose: [cifar10 | cifar100 | PACS | tinyimagenet | fourdataset]')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='[cuda | cpu]')
    parser.add_argument('--batch', type=int, default=64, 
                        help='batch size')
    parser.add_argument('--comms', type=int, default=60, 
                        help='communication rounds')
    parser.add_argument('--local_eps', type=int, default=5,
                        help='number of epochs in local clients training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--n_clients', type=int, default=80,help='the number of clients')
    parser.add_argument('--n_sampled_clients', type=int, default=10,
                        help='the number of sampled clients per round')
    parser.add_argument('--data_distribution', type=str, default='non_iid_dirichlet',
                        help='data split way to choose: [non_iid_dirichlet | manual_extreme_heterogeneity]')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='the level of non-iid data split')
    parser.add_argument('--n_dominated_class', type=int, default=1,
                        help='number of dominated class when applying manual_heterogeneity')
    parser.add_argument('--model_type', type=str, default='L2P',
                        help='choose model type you want: [L2P]')
    parser.add_argument('--prompt_method', type=str, default='shallow',
                        help='[shallow | deep]')
    parser.add_argument('--n_tokens', type=int, default=4,
                        help='number of tokens in prompt')
    parser.add_argument('--dominated_ratio', type=float, default=0.99,
                        help='ratio of dominated classs when applying manual_heterogeneity')
    parser.add_argument('--load_model', type=str, default= None,
                        help='the path of the pre-trained model')
    
    # L2P parameters
    parser.add_argument('--pool_size', type=int, default=20,
                        help='Define the prompt pool size in L2P model')
    parser.add_argument('--batchwise_prompt', type=bool, default=False,
                        help='Define L2P heuristic model if selecting top_k prompts in pool by batchwise')
    parser.add_argument('--nonpara_hidden', type=int, default=128,
                        help='Define the number of hidden neurons in Nonparametric aggregation method')
    parser.add_argument('--reduce_sim_scalar', type=float, default=0.5,
                        help='control reduce similarity in L2P model')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the trained model in the last epoch')
    parser.add_argument('--instance_label', type=int, default=1,
                        help='instance label')
    parser.add_argument('--query_model_type', type=str, default = "vit_cls",
                        help='Select query function, [vit_cls, deit_b16, convit_base]')
    parser.add_argument('--dp_mechanism', type=str, default='no_dp',
                        help='choose whether to apply differential privacy: [Laplace, Gaussian, AnalyticGaussian, Staircase, Prompt]')
    parser.add_argument('--eps', type=float, default=1.0,
                        help='Epsilon to apply for differential privacy: [1.0, 3.0, 5.0]')
    parser.add_argument('--defense', type=str, default='none',
                        choices=['none', 'perturb', 'sparse', 'mix_up'],
                        help='training-time defense to apply')
    parser.add_argument('--perturb_sigma', type=float, default=0.0,
                        help='std for Gaussian gradient perturbation')
    parser.add_argument('--sparse_ratio', type=float, default=0.0,
                        help='fraction of gradient entries to prune in sparse defense')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='beta(alpha, alpha) parameter for mixup')
    parser.add_argument('--data_augment', action='store_true',
                        help='enable train-time data augmentation')
    parser.add_argument('--sampling_proportion', type=float, default=1.0,
                        help='fraction of local mini-batches used per epoch in each client')
    parser.add_argument('--client_sampling_type', type=str, default='uniform',
                        choices=['uniform', 'poisson'],
                        help='client sampling scheme for cross-device FL')

    
    args = parser.parse_args()

    def _fmt(v):
        return str(v).replace('.', 'p')

    defense_tag = (
        f"def{args.defense}"
        f"_mix{_fmt(args.mixup_alpha)}"
        f"_spr{_fmt(args.sparse_ratio)}"
        f"_ptb{_fmt(args.perturb_sigma)}"
        f"_aug{int(args.data_augment)}"
        f"_sprop{_fmt(args.sampling_proportion)}"
        f"_cs{args.client_sampling_type}"
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # print(args.device)
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name())
    # print(args.batchwise_prompt)
    if args.n_clients > args.n_sampled_clients:
        scenrio_type = 'cross_devices'
    else:
        scenrio_type = 'cross_silo'
        
    norm_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    resize = 256
    centercrop_size = 224
    train_preprocess = build_data_transform(norm_stats, resize, centercrop_size, train_augment=args.data_augment)
    test_preprocess = build_data_transform(norm_stats, resize, centercrop_size, train_augment=False)
    class_mask = None
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./Dataset', train=True, download=True, transform=train_preprocess
        )
        testset = torchvision.datasets.CIFAR10(
            root='./Dataset', train=False, download=True, transform=test_preprocess
        )
        num_classes = 10
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='./Dataset', train=True, download=True, transform=train_preprocess
        )
        testset = torchvision.datasets.CIFAR100(
            root='./Dataset', train=False, download=True, transform=test_preprocess
        )
        num_classes = 100
    elif args.dataset == 'PACS':
        data_name = ['art_painting', 'cartoon', 'photo', 'sketch']
        trainset = [Pacs_reader('./Dataset/PACS', split=data_name[i], train=True, transform=train_preprocess, random_seed=i) for i in range(len(data_name))]
        testset = [Pacs_reader('./Dataset/PACS', split=data_name[i], train=False, transform=test_preprocess, random_seed=i) for i in range(len(data_name))]
        testset = Data.ConcatDataset(testset)
        num_classes = 28
    elif args.dataset == 'tinyimagenet':
        trainset = TinyImageNet_reader('./Dataset/tiny-imagenet-200/', train=True, transform=train_preprocess)
        testset = TinyImageNet_reader('./Dataset/tiny-imagenet-200/', train=False, transform=test_preprocess)
        num_classes = 200
    elif args.dataset == 'fourdataset':
        data_name = ['mnistm', 'fashion', 'cinic10', 'mmafedb']
        trainset, class_mask = four_dataset_reader([30000]*4, train=True, transform=train_preprocess)
        testset, _ = four_dataset_reader([2500]*4, train=False, transform=test_preprocess)
        num_classes = 37
        print(args.dataset)
    else:
        raise ValueError("Input dataset is not supported")
    print(class_mask)
    # setup global testing set & distributed dataset
    if args.dataset == 'fourdataset':
        testloader = [Data.DataLoader(testset[i], batch_size=args.batch, num_workers=2, shuffle=False) for i in range(len(testset))]
    else:
        testloader = Data.DataLoader(testset, batch_size=args.batch, num_workers=2, shuffle=False)
    if args.dataset == 'PACS' or args.dataset == 'fourdataset':
        data_partitioner = [DataPartitioner(trainset[i], args.n_clients//len(data_name)) for i in range(len(trainset))]
    else:
        data_partitioner = DataPartitioner(trainset, args.n_clients)
    
    if args.alpha == 0.01:
        seed = 688
    elif args.alpha == 0.05:
        seed = 805
    elif args.alpha == 0.1:
        seed = 871
    elif args.alpha == 0.2:
        seed = 459
    elif args.alpha == 0.3:
        seed = 429
    elif args.alpha == 0.4:
        seed = 3760
    elif args.alpha == 0.5:
        seed = 448
    else:
        raise ValueError(f"Unsupported alpha value for seed selection: {args.alpha}")
        
    #Get current time
    LATz = pytz.timezone("America/Los_Angeles") 
    timeInLA = datetime.now(LATz)
    
    if args.data_distribution == 'non_iid_dirichlet':
        if isinstance(data_partitioner, list):
            for i in range(len(data_partitioner)):
                if i==0:
                    data_partitioner[i].dirichlet_split_noniid(alpha=args.alpha, least_samples=32, manual_seed=seed)
                else:
                    data_partitioner[i].dirichlet_split_noniid(alpha=args.alpha, least_samples=32, manual_seed=data_partitioner[0].seed)
        else:
            data_partitioner.dirichlet_split_noniid(alpha=args.alpha, least_samples=32, manual_seed=seed)
        #setup log file for recording
        # npy_save_path = f"./output_record/loss_acc/info_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_alpha{args.alpha}_({args.instance_label})"
        # pool_save_path = f"./output_record/loss_acc/pool_record_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_alpha{args.alpha}_({args.instance_label})"
        # log_file_local_training = open(f"./output_record/log_output/local_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_alpha{args.alpha}_({args.instance_label}).txt", mode="w+", encoding="utf-8")
        # log_file_global_aggregation = open(f"./output_record/log_output/global_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_alpha{args.alpha}_({args.instance_label}).txt", 
        #                                    mode="a+", encoding="utf-8")
    elif args.data_distribution == 'manual_extreme_heterogeneity':
        if isinstance(data_partitioner, list):
            for i in range(len(data_partitioner)):
                data_partitioner[i].manual_allocating_noniid(args.n_dominated_class, args.dominated_ratio, 1.0)
        else:
            data_partitioner.manual_allocating_noniid(args.n_dominated_class, args.dominated_ratio, 1.0)
        #setup log file for recording
        # npy_save_path = f"./output_record/loss_acc/info_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_({args.instance_label})"
        # pool_save_path = f"./output_record/loss_acc/pool_record_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_({args.instance_label})"
        # log_file_local_training = open(f"./output_record/log_output/local_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_({args.instance_label}).txt", mode="w+", encoding="utf-8")
        # log_file_global_aggregation = open(f"./output_record/log_output/global_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_({args.instance_label}).txt", 
        #                                    mode="a+", encoding="utf-8")
    else:
        raise ValueError("Input data distribution is not supported")
    
   
    #prepare to record eval loss and accuracy
    eval_loss_record = list()
    eval_acc_record = list()
    eval_pool_size = None
    eval_pool = None
    client_pool = None
    
    # call FL algorithm and run
    if args.alg in ['fedavg','fedavg_nonpara']:
        # setup federated scenario
        if args.dataset == 'PACS' or args.dataset == 'fourdataset':
            all_clients_weights = np.concatenate([data_partitioner[i].get_all_client_weights() for i in range(len(data_partitioner))])
            distributed_dataloaders = [data_partitioner[i].get_distributed_data(batch_size=args.batch) for i in range(len(data_partitioner))]
            distributed_dataloaders = [distributed_dataloaders[i][j] for i in range(len(distributed_dataloaders)) for j in range(data_partitioner[i].n_clients)]
            print(len(distributed_dataloaders))
            fl_scen = FL_scenario(all_clients_weights,
                                  args.n_clients, args.n_sampled_clients,
                                  distributed_dataloaders)
        else:
            fl_scen = FL_scenario(data_partitioner.get_all_client_weights(),
                                  args.n_clients, args.n_sampled_clients,
                                  data_partitioner.get_distributed_data(batch_size=args.batch))
            
        print(args.model_type)
        if args.model_type == 'L2P':
            if args.query_model_type == 'vit_cls':
                server_model = to_device(L2P_ViT_B32(prompt_method=args.prompt_method,
                                                 batchwise_prompt=args.batchwise_prompt,
                                                 pool_size=args.pool_size,
                                                 top_k=args.n_tokens,
                                                 num_classes=num_classes,
                                                 candidate_key=None), args.device)
            elif args.query_model_type == 'cait_s24':
                server_model = to_device(
                    L2P_CaiT_S24(prompt_method=args.prompt_method,
                               batchwise_prompt=args.batchwise_prompt,
                               pool_size=args.pool_size,
                               top_k=args.n_tokens,
                               num_classes=num_classes,
                               candidate_key=None), args.device)
                
            elif args.query_model_type == 'deit_b16':
                server_model = to_device(
                    L2P_DeiT_B16(prompt_method=args.prompt_method,
                                batchwise_prompt=args.batchwise_prompt,
                                pool_size=args.pool_size,
                                top_k=args.n_tokens,
                                num_classes=num_classes,
                                candidate_key=None,
                                pretrained=True),
                    args.device
                )
            elif args.query_model_type == 'convit_base':
                server_model = to_device(
                    L2P_ConViT_Base(prompt_method=args.prompt_method,
                                batchwise_prompt=args.batchwise_prompt,
                                pool_size=args.pool_size,
                                top_k=args.n_tokens,
                                num_classes=num_classes,
                                candidate_key=None,
                                pretrained=True),
                    args.device
                )
            else:       
                raise ValueError("Model type is not supported")
            # server_model = to_device(L2P_ViT_B32(prompt_method=args.prompt_method,
            #                                      batchwise_prompt=args.batchwise_prompt,
            #                                      pool_size=args.pool_size,
            #                                      top_k=args.n_tokens,
            #                                      num_classes=num_classes,
            #                                      candidate_key=None, 
            #                                      query_model_type=args.query_model_type), args.device)
            eval_pool_size = list()
            eval_pool = list()
            client_pool = list()
            eval_pool_size.append(args.pool_size)
            eval_pool.append(server_model.pool.prompt.data.detach().cpu().numpy())
        else:
            raise ValueError("Model type is not supported")
        server_model.build_trainable_keys()
        #get FL algorithm
        if '_' in args.alg:
            associate_algo = args.alg.split('_')[0]
            algclass = get_algorithm(associate_algo)
        else:
            algclass = get_algorithm(args.alg)
        if args.alg == 'fedavg_nonpara':
            print('nonpara')
            if associate_algo == 'fedavg':
                algo = algclass(server_model=server_model,scenario=fl_scen,
                                loss_fun=nn.CrossEntropyLoss(),fed_method='nonparametric_aggregation',
                                nonpara_hidden=args.nonpara_hidden, class_mask=class_mask, device=args.device)
            else:
                raise ValueError("Algorithm is not supported")
        elif args.alg == 'fedavg':
            print('normal aggregation')
            algo = algclass(server_model=server_model,scenario=fl_scen,
                            loss_fun=nn.CrossEntropyLoss(),fed_method='aggregation', class_mask=class_mask, device=args.device)
        else:
            raise ValueError("Algorithm is not supported")
        if args.load_model is not None:
            load_model_trainable_part(algo.server_model, args.load_model)
            print(f"Model loaded from {args.load_model}\n")
            # 3) Build the algorithm (this should create client models)

            broadcast_like_communication(algo.server_model, algo.client_model, device=args.device)

        # print(f"*****Current Federated Learning Scenario is {scenrio_type}*****", file=log_file_global_aggregation)
        # print("-----------------------------------------------------------------------------------------", file=log_file_global_aggregation)
        # print(f"*****Current time is {timeInLA} ***** Total communication rounds are {args.comms} *****", file=log_file_global_aggregation)
        # print("-----------------------------------------------------------------------------------------", file=log_file_global_aggregation)
        # print(f"Learning Rate: {args.lr}; Batch size: {args.batch}", file=log_file_global_aggregation)
        best_model_path = f"./output_record/saved_model/saved_model_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_model{args.query_model_type}_{args.data_distribution}_alpha{args.alpha}_{args.instance_label}_{defense_tag}_init.pt"
        save_model_trainable_part(algo.server_model, best_model_path)
        print("\n✅ Save init model\n")
        for comm_round in trange(args.comms):
            # client train
            algo.client_train(comm_round=comm_round, epochs=args.local_eps, lr=args.lr, 
                              dp_mechanism='no_dp', eps=args.eps,
                            frac=args.n_sampled_clients/args.n_clients, reduce_sim_scalar=args.reduce_sim_scalar,
                            print_output=True, defense=args.defense, mixup_alpha=args.mixup_alpha,
                            sparse_ratio=args.sparse_ratio, perturb_sigma=args.perturb_sigma,
                            sampling_proportion=args.sampling_proportion,
                            client_sampling_type=args.client_sampling_type)
            # record client prompt
            if args.model_type == 'L2P':
                for i in range(len(algo.selected_client_index)):
                    client_pool.append(algo.client_model[i].pool.prompt.data.detach().cpu().numpy())
            # sever aggregate
            algo.server_aggre()
            # if args.attack:
            #     algo.adversarial_inject(Data.DataLoader(trainset, batch_size=1, num_workers=1, shuffle=True), args.n_sampled_clients, args.n_tokens, args.query_model_type)
                
            # print(f'--------------------------Round {comm_round+1} complete----------------------------',
            #     file=log_file_local_training)
            # evaluate
            if args.model_type == 'L2P':
                eval_loss, eval_acc, current_pool_size = algo.server_eval(testloader, comm_round)
                eval_pool_size.append(current_pool_size)
                eval_pool.append(algo.server_model.pool.prompt.data.detach().cpu().numpy())
            else:
                eval_loss, eval_acc = algo.server_eval(testloader, comm_round)
            eval_loss_record.append(eval_loss)
            eval_acc_record.append(eval_acc)
            best_model_path = f"./output_record/saved_model/saved_model_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_model{args.query_model_type}_{args.data_distribution}_alpha{args.alpha}_{args.instance_label}_{defense_tag}_round_{comm_round+1}_acc_{eval_acc:.4f}.pt"
            if args.save_model and comm_round%59==0:
                save_model_trainable_part(algo.server_model, best_model_path)
                print(f"\n✅ New last model saved to {best_model_path} at round {comm_round+1} with acc: {eval_acc:.4f}\n")
        min_index = np.argmin(np.array(eval_loss_record))
        print(f'*****Final accuracy with minimal eval_loss: {eval_acc_record[min_index]}*****')
            
    else:
        raise ValueError("Algorithm is not supported")
    #save record to npy files
    # save_eval_npy_file(eval_loss_record=eval_loss_record, eval_acc_record=eval_acc_record, 
    #                    eval_pool_size=eval_pool_size, file_path=npy_save_path+'.npy')
    # if args.model_type == 'L2P':
    #     np.save(pool_save_path+'.npy', np.concatenate(eval_pool, axis=0))
    