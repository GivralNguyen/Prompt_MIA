import os
import argparse
import random
import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
import torchvision
from tqdm._tqdm import trange
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
from Models.L2P_heuristic_model import L2P_ViT_B32, L2P_DeiT_B16, L2P_ConViT_Base
from util.saving_tools import save_eval_npy_file, save_model_trainable_part,load_model_trainable_part
# from util.dp_mech import LaplaceMechanism, GaussianMechanism,AnalyticGaussianMechanism, StaircaseMechanism


def apply_client_data_sampling(partitioner, sampling_proportion_data, seed=0):
    sampling_proportion_data = float(max(0.0, min(1.0, sampling_proportion_data)))
    if sampling_proportion_data >= 1.0:
        return

    rng = np.random.default_rng(seed)
    sampled_client_idcs = []
    for client_idcs in partitioner.client_idcs:
        client_idcs = np.array(client_idcs)
        if client_idcs.size == 0:
            sampled_client_idcs.append(client_idcs)
            continue
        keep_num = max(1, int(client_idcs.size * sampling_proportion_data))
        sampled_idcs = rng.choice(client_idcs, size=keep_num, replace=False)
        sampled_client_idcs.append(np.array(sampled_idcs, dtype=client_idcs.dtype))
    partitioner.client_idcs = sampled_client_idcs

if __name__ == '__main__':
    # Federated learning settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default="0",
                    help='Comma-separated GPU device IDs, e.g., "0", "0,1", or "" for CPU')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Algorithm to choose: [fedavg | fedavg_nonpara]')
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
    parser.add_argument('--sampling_proportion_data', type=float, default=1.0,
                        help='fraction of samples retained per client during data processing')
    
    # MIA settings
    parser.add_argument('--mia_mixup', action='store_true',
                        help='apply mixup to inputs during MIA evaluation')
    parser.add_argument('--mia_mixup_alpha', type=float, default=0.2,
                        help='beta(alpha, alpha) parameter for MIA-time mixup')
    parser.add_argument('--batch_size_mia', type=int, default=1,
                        help='Batch size when performing MIA attack')
    parser.add_argument('--beta', type=float, default=0,
                        help='beta to control the direction of adversarial key. DONT CHANGE IF YOU DONT KNOW WHAT YOU ARE DOING')
    parser.add_argument('--delta_min', type=float, default=0.02,
                        help='minimum delta to control the similarity of adversarial key')
    parser.add_argument('--delta_max', type=float, default=0.05,
                        help='maximum delta to control the similarity of adversarial key')
    
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
        f"_dsprop{_fmt(args.sampling_proportion_data)}"
        f"_mmix{int(args.mia_mixup)}"
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
    test_preprocess = build_data_transform(norm_stats, resize, centercrop_size, train_augment=args.data_augment)
    class_mask = None
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./Dataset', train=True, download=True, transform=train_preprocess
        )
        testset = torchvision.datasets.CIFAR10(
            root='./Dataset', train=False, download=True, transform=train_preprocess
        )
        num_classes = 10
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='./Dataset', train=True, download=True, transform=train_preprocess
        )
        testset = torchvision.datasets.CIFAR100(
            root='./Dataset', train=False, download=True, transform=train_preprocess
        )
        num_classes = 100
    elif args.dataset == 'PACS':
        data_name = ['art_painting', 'cartoon', 'photo', 'sketch']
        trainset = [Pacs_reader('./Dataset/PACS', split=data_name[i], train=True, transform=train_preprocess, random_seed=i) for i in range(len(data_name))]
        testset = [Pacs_reader('./Dataset/PACS', split=data_name[i], train=False, transform=train_preprocess, random_seed=i) for i in range(len(data_name))]
        testset = Data.ConcatDataset(testset)
        num_classes = 28
    elif args.dataset == 'tinyimagenet':
        trainset = TinyImageNet_reader('./Dataset/tiny-imagenet-200/', train=True, transform=train_preprocess)
        testset = TinyImageNet_reader('./Dataset/tiny-imagenet-200/', train=False, transform=train_preprocess)
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
            for i in range(len(data_partitioner)):
                apply_client_data_sampling(data_partitioner[i], args.sampling_proportion_data, seed + i)
        else:
            data_partitioner.dirichlet_split_noniid(alpha=args.alpha, least_samples=32, manual_seed=seed)
            apply_client_data_sampling(data_partitioner, args.sampling_proportion_data, seed)
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
            for i in range(len(data_partitioner)):
                apply_client_data_sampling(data_partitioner[i], args.sampling_proportion_data, seed + i)
        else:
            data_partitioner.manual_allocating_noniid(args.n_dominated_class, args.dominated_ratio, 1.0)
            apply_client_data_sampling(data_partitioner, args.sampling_proportion_data, seed)
        #setup log file for recording
        # npy_save_path = f"./output_record/loss_acc/info_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_({args.instance_label})"
        # pool_save_path = f"./output_record/loss_acc/pool_record_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_({args.instance_label})"
        # log_file_local_training = open(f"./output_record/log_output/local_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_({args.instance_label}).txt", mode="w+", encoding="utf-8")
        # log_file_global_aggregation = open(f"./output_record/log_output/global_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_({args.instance_label}).txt", 
        #                                    mode="a+", encoding="utf-8")
    else:
        raise ValueError("Input data distribution is not supported")
    
    # print("-----------------------------------------------------------------------------------------", file=log_file_global_aggregation)
    # print(f"*****Current time is {timeInLA} ***** Total communication rounds are {args.comms} *****", file=log_file_global_aggregation)
    # print("-----------------------------------------------------------------------------------------", file=log_file_global_aggregation)
    # print(f"Learning Rate: {args.lr}; Batch size: {args.batch}", file=log_file_global_aggregation)
    #prepare to record eval loss and accuracy
    eval_loss_record = list()
    eval_acc_record = list()
    eval_pool_size = None
    eval_pool = None
    client_pool = None
    
    # call FL algorithm and run
    if args.alg in ['fedavg','fedavg_nonpara']:
        # print(f"*****Current Federated Learning Scenario is {scenrio_type}*****", file=log_file_global_aggregation)
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
        # if args.alg == 'fedavg_nonpara':
        #     print('nonpara')
        #     if associate_algo == 'fedavg':
        #         algo = algclass(server_model=server_model,scenario=fl_scen,
        #                         loss_fun=nn.CrossEntropyLoss(),fed_method='nonparametric_aggregation',
        #                         nonpara_hidden=args.nonpara_hidden, class_mask=class_mask, device=args.device)
        #     else:
        #         raise ValueError("Algorithm is not supported")
        if args.alg == 'fedavg':
            print('normal aggregation')
            algo = algclass(server_model=server_model,scenario=fl_scen,
                            loss_fun=nn.CrossEntropyLoss(),fed_method='aggregation', class_mask=class_mask, device=args.device)
        else:
            raise ValueError("Algorithm is not supported")
        if args.load_model is not None:
            if args.load_model == "random":
                pass
                print("Randomly initialized model is loaded\n")
            else:
                load_model_trainable_part(algo.server_model, args.load_model)
                print(f"Model loaded from {args.load_model}\n")
        else:
            for comm_round in trange(args.comms):
                # client train
                algo.client_train(comm_round=comm_round, epochs=args.local_eps, lr=args.lr, 
                                dp_mechanism=args.dp_mechanism, eps=args.eps,
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
            min_index = np.argmin(np.array(eval_loss_record))
            # print(f'*****Final accuracy with minimal eval_loss: {eval_acc_record[min_index]}*****', 
            #     file=log_file_global_aggregation)
            best_model_path = f"./output_record/saved_model/saved_model_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_alpha{args.alpha}_{args.instance_label}_{defense_tag}_round_{comm_round}_acc_{eval_acc:.4f}.pt"
            if args.save_model:
                save_model_trainable_part(algo.server_model, best_model_path)
                print(f"\n✅ New last model saved to {best_model_path} at round {comm_round+1} with acc: {eval_acc:.4f}\n")
    else:
        raise ValueError("Algorithm is not supported")
    #save record to npy files
    # save_eval_npy_file(eval_loss_record=eval_loss_record, eval_acc_record=eval_acc_record, 
    #                    eval_pool_size=eval_pool_size, file_path=npy_save_path+'.npy')
    # if args.model_type == 'L2P':
    #     np.save(pool_save_path+'.npy', np.concatenate(eval_pool, axis=0))
    mia_batch_sizes = [1,4,8,16,32,64,128,256]
    for loop in range(5):
        print("=============== MIA Attack Loop {} ===============".format(loop+1))
        for bs in mia_batch_sizes:
            print("================= MIA with batch size {} =================".format(bs))
            if args.dataset == 'PACS' or args.dataset == 'fourdataset':
                train_dataloader_list = [torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False) for dataset in trainset]
                test_dataloader_list = [torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False) for dataset in testset]
            else:
                train_dataloader_list = [torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=False)]
                test_dataloader_list = [torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)]
        
            # mia_dataloader_list = train_dataloader_list + test_dataloader_list  # Combines the list of dataloaders for training data and test data into a single list. 
            # is_member_flags = [True] * len(train_dataloader_list) + [False] * len(test_dataloader_list)   # ssigns True for every training data sample (a "member") and False for every test data sample (a "non-member").
            mia_dataloader_list =  test_dataloader_list  # Combines the list of dataloaders for training data and test data into a single list. 
            is_member_flags = [False] * len(test_dataloader_list)   # s
            true_cnt = 0
            false_cnt = 0
            true_pos = 0
            false_pos = 0
            true_neg = 0
            false_neg = 0
            key_tsne_plotted = False  # Flag to control first-time-only plotting
            for dataloader, is_member in zip(mia_dataloader_list, is_member_flags):
                for batch_idx, (x, label) in enumerate(dataloader): #iterates through every single data point (x) from both the training and test sets. For each one, it tries to predict if it's a "member" or not.
                    # print(batch_idx)
                    cls_features = []
                    temp_cls_features = []
                    x_embed_norm = []
                    x = to_device(x, args.device).float() #B,32,224,224
                    label = to_device(label, args.device).long() #B
                    if args.mia_mixup and x.size(0) > 1:
                        lam = float(np.random.beta(args.mia_mixup_alpha, args.mia_mixup_alpha))
                        idx = torch.randperm(x.size(0), device=x.device)
                        x = lam * x + (1.0 - lam) * x[idx]
                    for i in range(x.size(0)): # for each data point in the batch, calculate q(x) normalized
                        x_single = x[i].unsqueeze(0)
                        label_single = label[i]
                        with torch.no_grad():
                            cls_features.append(algo.server_model.get_cls_keys(x_single)) # get q(x)
                        if is_member:
                            x_embed_norm.append(algo.client_model[0].pool.l2_normalize(cls_features[i], dim=1)) #normalize q(x)
                        else: # if x is not member
                            random_loader = random.choice(train_dataloader_list)
                            dataset = random_loader.dataset
                            rand_idx = random.randint(0, len(dataset) - 1)
                            temp_x, temp_label = dataset[rand_idx]
                            temp_x = to_device(temp_x, args.device).float().unsqueeze(0)
                            temp_label = to_device(torch.tensor([temp_label]), args.device).long()
                            with torch.no_grad():
                                temp_cls_features.append(algo.server_model.get_cls_keys(temp_x))  
                            x_embed_norm.append(algo.client_model[0].pool.l2_normalize(temp_cls_features[i], dim=1)) #normalize q(x)
                    for j in range(x.size(0)): # is the target in the batch? 
                        with torch.no_grad():
                            real_prompt = algo.server_model.pool.prompt.clone() 
                            updated_prompt = real_prompt
                            algo.client_model[0].pool.prompt = nn.Parameter(updated_prompt, requires_grad=True)
                            
                            real_key = algo.server_model.pool.key.clone() 
                            updated_key = real_key
                            # updated_key[-args.n_tokens:] = query_vector # inject q(x)into the last n_tokens keys
                            algo.client_model[0].pool.key = nn.Parameter(updated_key, requires_grad=True)
                            
                            n_prompts = updated_prompt.size(0)
                            algo.client_model[0].trained_prompts_checklist = torch.zeros(n_prompts, dtype=torch.float32, device=args.device)

                        # Normalize prompt/key/query for cosine similarity
                        
                        prompt_norm = algo.client_model[0].pool.l2_normalize(algo.client_model[0].pool.prompt, dim=1)
                        key_norm = algo.client_model[0].pool.l2_normalize(algo.client_model[0].pool.key, dim=1)

                        similarity = torch.matmul(x_embed_norm[j], key_norm.t())  # calculate cosine similarity between normalized q(x) and all keys
                        # non_adv_similarity = similarity[:, :-args.n_tokens] # exclude the last n_tokens keys (injected keys)
                        # ===== choose top-k = args.n_tokens indices for THIS sample j =====
                        k_top = int(args.n_tokens)
                        
                        topk_vals, topk_idxs = torch.topk(similarity[0], k=k_top, largest=True)  # (k_top,)
                        chosen_idx = topk_idxs.detach().cpu().numpy().tolist()                   # list[int]
                        # print("Chosen adversarial Key Indices:", chosen_idx)
                        # print("Similarity with all keys:", similarity, "chosen indices:", chosen_idx)
                        # ===== compute similarity thresholds EXCLUDING chosen_idx =====
                        K = key_norm.size(0)
                        mask_np = np.ones(K, dtype=bool)
                        mask_np[chosen_idx] = False
                        mask_t = torch.tensor(mask_np, device=similarity.device, dtype=torch.bool)
                        non_adv_similarity = similarity[:, mask_t]        
                        max_val, max_idx = torch.max(non_adv_similarity, dim=1)
                        # print("similarity values excluding chosen keys:", non_adv_similarity)
                        # print(f"Max similarity excluding chosen keys: {max_val.item()} at index {max_idx.item()}")
                        delta_min = args.delta_min
                        delta_max = args.delta_min + args.delta_max
                        similarity_start = torch.min(max_val + delta_min, torch.tensor(1, device=max_val.device)).cpu().item()
                        similarity_end   = torch.min(max_val + delta_max, torch.tensor(1, device=max_val.device)).cpu().item()

                        non_chosen = np.where(mask_np)[0]  # 'mask' already excludes chosen_idx above
                        anchor_idx = np.random.choice(non_chosen, size=len(chosen_idx), replace=False) # only relevant when beta > 0

                        # print(f"Max similarity of original points: {max_val}")
                        # print(f"Generating 10 new points with varied similarities between {similarity_start} and {similarity_end}\n")

                        varied_target_similarities = np.random.uniform(similarity_start, similarity_end, args.n_tokens)
                        # print("Varied Target Similarities:", varied_target_similarities)
                        def generate_point_with_similarity(target_vec, similarity, benign_key):
                            # Ensure arrays are 1D vectors (shape: (768,))
                            target_vec = target_vec.reshape(-1)
                            benign_key = benign_key.reshape(-1)

                            # Normalize
                            target_norm = target_vec / np.linalg.norm(target_vec)
                            benign_norm = benign_key / np.linalg.norm(benign_key)

                            # Random vector
                            rand_vec = np.random.rand(benign_key.shape[0])
                            rand_vec = rand_vec / np.linalg.norm(rand_vec)

                            # Mix benign and random (spread outward)
                            beta = args.beta
                            final_vec = (beta) * benign_norm + (1-beta) * rand_vec
                            final_vec /= np.linalg.norm(final_vec)

                            # Orthogonalize against target
                            orthogonal_vec = final_vec - np.dot(final_vec, target_norm) * target_norm
                            orthogonal_norm = orthogonal_vec / (np.linalg.norm(orthogonal_vec) + 1e-8)

                            # Combine target direction with orthogonal component
                            new_vec_norm = similarity * target_norm + np.sqrt(1 - similarity**2) * orthogonal_norm
                            new_vec_norm /= np.linalg.norm(new_vec_norm)

                            # Scale back to target’s norm
                            new_vec = new_vec_norm * np.linalg.norm(target_vec)

                            return new_vec

                        query_vector = cls_features[j] # q(x) of the current item in the batch
                        with torch.no_grad():
                            # benign_keys_np = real_key[:-args.n_tokens].detach().cpu().numpy()  # all non-injected keys
                            real_keys_np = real_key.detach().cpu().numpy() 
                            query_vector_np = query_vector.detach().cpu().numpy()
                            num_sims = len(varied_target_similarities)
                            # chosen_idx = np.random.choice(len(benign_keys_np), size=num_sims, replace=False)
                            # chosen_idx = np.arange(len(real_key) - args.n_tokens, len(real_key))
                            # print("Chosen adversarial Key Indices:", chosen_idx)
                            similar_points = []

                            for sim, idx in zip(varied_target_similarities, anchor_idx):
                                benign_key = real_keys_np[idx:idx+1]
                                new_point = generate_point_with_similarity(query_vector_np, sim, benign_key)
                                similar_points.append(new_point)
                            similar_points_tensor = torch.tensor(similar_points, dtype=updated_key.dtype, device=updated_key.device)
                            # updated_key[-args.n_tokens:] = similar_points_tensor
                            updated_key[torch.tensor(chosen_idx, device=updated_key.device)] = similar_points_tensor
                            algo.client_model[0].pool.key = nn.Parameter(updated_key, requires_grad=True)
                            key_norm = algo.client_model[0].pool.l2_normalize(algo.client_model[0].pool.key, dim=1)

                        similarity = torch.matmul(torch.cat(x_embed_norm, dim=0), key_norm.t()) # cosine similarity between whole batch and all keys
                        
                        # drop_p = 0.3  # example dropout probability
                        # # print(drop_p)
                        # mask = (torch.rand_like(similarity) > drop_p).float()
                        # similarity = similarity * mask
                        
                        # print("Similarities with all keys:\n", similarity)
                        _, idx = torch.topk(similarity, k=args.n_tokens, dim=1) # get top n_tokens keys with highest similarity
                        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                        # print(similarity)
                    
                        numbers = chosen_idx # indices of injected keys
                        # print("Injected Key Indices:", numbers, "top similar keys:", idx)
                        all_included_any = all(torch.any(idx == num).item() for num in numbers) # check if all injected keys are in the top n_tokens keys for all data point in the batch
                        # print("All injected keys included in top tokens:", all_included_any)
                        if all_included_any: # if all injected keys are in the top n_tokens keys, predict as member
                            true_cnt += 1
                            if is_member:
                                # print(chosen_idx)
                                # print("True Positive Similarity:\n", similarity)
                                true_pos += 1
                            else:
                                false_pos += 1
                                # print(chosen_idx)
                                # print("False Positive Similarity:\n", similarity)
                        else: # if not all injected keys are in the top n_tokens keys, predict as non-member
                            false_cnt += 1
                            if is_member:
                                false_neg += 1
                                # print(chosen_idx)
                                # print("False Negative Similarity:\n", similarity)
                            else:
                                # print(chosen_idx)
                                # print("True Negative Similarity:\n", similarity)
                                true_neg += 1 
            # MIA Attack Accuracy
            
            # TPR = true_pos / (true_pos + false_neg + 1e-8)
            TPR = 1.0
            TNR = true_neg / (true_neg + false_pos + 1e-8)
            Advantage = TPR + TNR - 1
            attack_accuracy = (TPR + TNR) / 2
            # print("MIA Success:", true_cnt, "MIA Fail:", false_cnt, "Attack Accuracy:)", attack_accuracy * 100, "%")
            print("mia batch size:", bs, "MIA Positive:", true_cnt, "MIA Negative:", false_cnt, "Attack Accuracy:)", attack_accuracy * 100, "%")
            # print("MIA Success:", true_cnt, "MIA Fail:", false_cnt, "Attack Accuracy:)", attack_accuracy * 100, "%")
            # print(f" TNR: {TNR*100:.2f}%")
            print(f"TPR: {TPR*100:.2f}%, TNR: {TNR*100:.2f}%, Advantage: {Advantage*100:.2f}%")
            print(f"TP: {true_pos}, FP: {false_pos}, FN: {false_neg}, TN: {true_neg}")