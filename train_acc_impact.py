import os
import argparse
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
from Models.L2P_heuristic_model import L2P_ViT_B32, L2P_CaiT_S24, L2P_DeiT_B16, L2P_ConViT_Base
from util.saving_tools import save_model_trainable_part, load_model_trainable_part


def broadcast_like_communication(server_model, client_models, device='cuda'):
    client_num = len(client_models)

    for key in server_model.trainable_keys:
        if 'prompt' not in key and 'key' not in key:
            temp = server_model.state_dict()[key]
            with torch.no_grad():
                for client_idx in range(client_num):
                    client_models[client_idx].state_dict()[key].data.copy_(temp)
        elif 'prompt' in key and 'key' not in key:
            if key.replace('prompt', 'key') not in server_model.state_dict():
                raise KeyError(f"Cannot find key tensor for {key}")

            if hasattr(server_model, 'trained_prompts_checklist'):
                with torch.no_grad():
                    server_prompt = server_model.pool.prompt.detach()
                    server_key = server_model.pool.key.detach()
                    server_model.pool.prompt = nn.Parameter(server_prompt, requires_grad=True)
                    server_model.pool.key = nn.Parameter(server_key, requires_grad=True)
                    for client_idx in range(client_num):
                        client_models[client_idx].pool.prompt = nn.Parameter(server_prompt.clone(), requires_grad=True)
                        client_models[client_idx].pool.key = nn.Parameter(server_key.clone(), requires_grad=True)
            else:
                with torch.no_grad():
                    server_prompt = server_model.prompt_embeddings.detach()
                    server_model.prompt_embeddings = nn.Parameter(server_prompt, requires_grad=True)
                    for client_idx in range(client_num):
                        client_models[client_idx].prompt_embeddings = nn.Parameter(server_prompt.clone(), requires_grad=True)
        else:
            temp = server_model.state_dict()[key]
            with torch.no_grad():
                for client_idx in range(client_num):
                    client_models[client_idx].state_dict()[key].data.copy_(temp)


def _fmt(value):
    return str(value).replace('.', 'p')


def _resolve_dataset_sample(dataset, sample_index, dataset_index=0):
    if isinstance(dataset, list):
        if dataset_index < 0 or dataset_index >= len(dataset):
            raise IndexError(f"inject_dataset_idx {dataset_index} is out of range for {len(dataset)} datasets")
        target_dataset = dataset[dataset_index]
    else:
        target_dataset = dataset

    if len(target_dataset) == 0:
        raise ValueError('Cannot inject from an empty dataset')

    sample_index = int(sample_index) % len(target_dataset)
    sample, _ = target_dataset[sample_index]
    return sample.unsqueeze(0)


def _resolve_inject_count(pool_size, inject_rate, fallback_tokens):
    if pool_size <= 0:
        raise ValueError('Prompt pool must contain at least one prompt')

    requested = int(round(pool_size * inject_rate))
    if requested <= 0:
        requested = int(fallback_tokens)
    requested = max(1, requested)
    if pool_size > 1:
        requested = min(requested, pool_size - 1)
    else:
        requested = 1
    return requested


def _generate_point_with_similarity(target_vec, similarity, benign_key, beta, rng):
    target_vec = np.asarray(target_vec).reshape(-1)
    benign_key = np.asarray(benign_key).reshape(-1)

    target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-8)
    benign_norm = benign_key / (np.linalg.norm(benign_key) + 1e-8)

    rand_vec = rng.random(benign_key.shape[0])
    rand_vec = rand_vec / (np.linalg.norm(rand_vec) + 1e-8)

    final_vec = beta * benign_norm + (1 - beta) * rand_vec
    final_vec = final_vec / (np.linalg.norm(final_vec) + 1e-8)

    orthogonal_vec = final_vec - np.dot(final_vec, target_norm) * target_norm
    orthogonal_norm = orthogonal_vec / (np.linalg.norm(orthogonal_vec) + 1e-8)

    new_vec_norm = similarity * target_norm + np.sqrt(max(0.0, 1 - similarity ** 2)) * orthogonal_norm
    new_vec_norm = new_vec_norm / (np.linalg.norm(new_vec_norm) + 1e-8)
    return new_vec_norm * (np.linalg.norm(target_vec) + 1e-8)


def inject_stealth_prompt_pool(server_model, sample_batch, args):
    if not hasattr(server_model, 'pool') or not hasattr(server_model.pool, 'key'):
        raise ValueError('Stealth injection requires an L2P-style model with a prompt pool')

    pool_size = server_model.pool.key.shape[0]
    inject_count = _resolve_inject_count(pool_size, args.inject_rate, args.n_tokens)
    rng = np.random.default_rng(args.inject_seed)
    sample_batch = to_device(sample_batch, args.device).float()

    with torch.no_grad():
        real_prompt = server_model.pool.prompt.detach().clone()
        real_key = server_model.pool.key.detach().clone()

        cls_features = server_model.get_cls_keys(sample_batch)
        query_norm = server_model.pool.l2_normalize(cls_features, dim=1)
        key_norm = server_model.pool.l2_normalize(real_key, dim=1)
        similarity = torch.matmul(query_norm, key_norm.t())

        _, chosen_tensor = torch.topk(similarity[0], k=inject_count, largest=True)
        chosen_idx = chosen_tensor.detach().cpu().numpy()

        if pool_size == 1:
            anchor_idx = np.array([0], dtype=np.int64)
            similarity_start = max(0.0, min(1.0, args.delta_min))
            similarity_end = max(similarity_start, min(1.0, args.delta_min + args.delta_max))
        else:
            mask_np = np.ones(pool_size, dtype=bool)
            mask_np[chosen_idx] = False
            mask_tensor = torch.tensor(mask_np, device=similarity.device, dtype=torch.bool)
            non_adv_similarity = similarity[:, mask_tensor]
            max_val, _ = torch.max(non_adv_similarity, dim=1)
            similarity_start = torch.minimum(
                max_val + args.delta_min,
                torch.tensor(1.0, device=max_val.device)
            ).cpu().item()
            similarity_end = torch.minimum(
                max_val + args.delta_min + args.delta_max,
                torch.tensor(1.0, device=max_val.device)
            ).cpu().item()

            non_chosen = np.where(mask_np)[0]
            replace = len(non_chosen) < len(chosen_idx)
            anchor_idx = rng.choice(non_chosen, size=len(chosen_idx), replace=replace)

        varied_target_similarities = rng.uniform(similarity_start, similarity_end, size=len(chosen_idx))
        real_keys_np = real_key.detach().cpu().numpy()
        query_vector_np = cls_features[0].detach().cpu().numpy()

        similar_points = []
        for target_similarity, benign_idx in zip(varied_target_similarities, anchor_idx):
            benign_key = real_keys_np[benign_idx:benign_idx + 1]
            similar_points.append(
                _generate_point_with_similarity(query_vector_np, target_similarity, benign_key, args.beta, rng)
            )

        similar_points_tensor = torch.tensor(
            np.stack(similar_points, axis=0),
            dtype=real_key.dtype,
            device=real_key.device
        )
        updated_key = real_key.clone()
        updated_key[torch.tensor(chosen_idx, device=updated_key.device)] = similar_points_tensor

        server_model.pool.prompt = nn.Parameter(real_prompt, requires_grad=True)
        server_model.pool.key = nn.Parameter(updated_key, requires_grad=True)
        if hasattr(server_model, 'trained_prompts_checklist'):
            server_model.trained_prompts_checklist = torch.zeros(
                real_prompt.size(0), dtype=torch.float32, device=real_prompt.device
            )

    return chosen_idx.tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated GPU device IDs, e.g., "0", "0,1", or "" for CPU')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Algorithm to choose: [fedavg]')
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
    parser.add_argument('--n_clients', type=int, default=80, help='the number of clients')
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
    parser.add_argument('--load_model', type=str, default=None,
                        help='the path of the pre-trained model')

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
    parser.add_argument('--query_model_type', type=str, default='vit_cls',
                        help='Select query function, [vit_cls, cait_s24, deit_b16, convit_base]')
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

    parser.add_argument('--inject_epoch', type=int, default=30,
                        help='communication round at which the stealth injection is applied; set <= 0 to disable')
    parser.add_argument('--inject_rate', type=float, default=0.2,
                        help='fraction of prompt-pool slots to overwrite during the stealth injection')
    parser.add_argument('--inject_sample_idx', type=int, default=0,
                        help='sample index used to build the injected stealth key pattern')
    parser.add_argument('--inject_dataset_idx', type=int, default=0,
                        help='dataset index for list-based datasets such as PACS or fourdataset')
    parser.add_argument('--inject_seed', type=int, default=0,
                        help='random seed for constructing stealth keys')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='beta to control the direction of adversarial key')
    parser.add_argument('--delta_min', type=float, default=0.02,
                        help='minimum delta to control the similarity of adversarial key')
    parser.add_argument('--delta_max', type=float, default=0.05,
                        help='maximum delta offset to control the similarity of adversarial key')

    args = parser.parse_args()

    defense_tag = (
        f"def{args.defense}"
        f"_mix{_fmt(args.mixup_alpha)}"
        f"_spr{_fmt(args.sparse_ratio)}"
        f"_ptb{_fmt(args.perturb_sigma)}"
        f"_aug{int(args.data_augment)}"
        f"_sprop{_fmt(args.sampling_proportion)}"
        f"_cs{args.client_sampling_type}"
    )
    inject_tag = (
        f"injE{args.inject_epoch}"
        f"_injR{_fmt(args.inject_rate)}"
        f"_injS{args.inject_sample_idx}"
        f"_injD{args.inject_dataset_idx}"
        f"_injSeed{args.inject_seed}"
        f"_beta{_fmt(args.beta)}"
        f"_dmin{_fmt(args.delta_min)}"
        f"_dmax{_fmt(args.delta_max)}"
    )

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
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
        trainset = [
            Pacs_reader('./Dataset/PACS', split=data_name[i], train=True, transform=train_preprocess, random_seed=i)
            for i in range(len(data_name))
        ]
        testset = [
            Pacs_reader('./Dataset/PACS', split=data_name[i], train=False, transform=test_preprocess, random_seed=i)
            for i in range(len(data_name))
        ]
        testset = Data.ConcatDataset(testset)
        num_classes = 28
    elif args.dataset == 'tinyimagenet':
        trainset = TinyImageNet_reader('./Dataset/tiny-imagenet-200/', train=True, transform=train_preprocess)
        testset = TinyImageNet_reader('./Dataset/tiny-imagenet-200/', train=False, transform=test_preprocess)
        num_classes = 200
    elif args.dataset == 'fourdataset':
        data_name = ['mnistm', 'fashion', 'cinic10', 'mmafedb']
        trainset, class_mask = four_dataset_reader([30000] * 4, train=True, transform=train_preprocess)
        testset, _ = four_dataset_reader([2500] * 4, train=False, transform=test_preprocess)
        num_classes = 37
        print(args.dataset)
    else:
        raise ValueError('Input dataset is not supported')

    print(class_mask)
    if args.dataset == 'fourdataset':
        testloader = [Data.DataLoader(testset[i], batch_size=args.batch, num_workers=2, shuffle=False) for i in range(len(testset))]
    else:
        testloader = Data.DataLoader(testset, batch_size=args.batch, num_workers=2, shuffle=False)

    if args.dataset in ['PACS', 'fourdataset']:
        data_partitioner = [DataPartitioner(trainset[i], args.n_clients // len(data_name)) for i in range(len(trainset))]
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
        raise ValueError(f'Unsupported alpha value for seed selection: {args.alpha}')

    latz = pytz.timezone('America/Los_Angeles')
    time_in_la = datetime.now(latz)

    if args.data_distribution == 'non_iid_dirichlet':
        if isinstance(data_partitioner, list):
            for i in range(len(data_partitioner)):
                if i == 0:
                    data_partitioner[i].dirichlet_split_noniid(alpha=args.alpha, least_samples=32, manual_seed=seed)
                else:
                    data_partitioner[i].dirichlet_split_noniid(
                        alpha=args.alpha,
                        least_samples=32,
                        manual_seed=data_partitioner[0].seed
                    )
        else:
            data_partitioner.dirichlet_split_noniid(alpha=args.alpha, least_samples=32, manual_seed=seed)
    elif args.data_distribution == 'manual_extreme_heterogeneity':
        if isinstance(data_partitioner, list):
            for i in range(len(data_partitioner)):
                data_partitioner[i].manual_allocating_noniid(args.n_dominated_class, args.dominated_ratio, 1.0)
        else:
            data_partitioner.manual_allocating_noniid(args.n_dominated_class, args.dominated_ratio, 1.0)
    else:
        raise ValueError('Input data distribution is not supported')

    print('-----------------------------------------------------------------------------------------')
    print(f'Current time is {time_in_la} | total communication rounds: {args.comms}')
    print('-----------------------------------------------------------------------------------------')

    eval_loss_record = []
    eval_acc_record = []
    eval_pool_size = None
    eval_pool = None
    client_pool = None

    if args.alg not in ['fedavg', 'fedavg_nonpara']:
        raise ValueError('Algorithm is not supported')

    if args.dataset in ['PACS', 'fourdataset']:
        all_clients_weights = np.concatenate([data_partitioner[i].get_all_client_weights() for i in range(len(data_partitioner))])
        distributed_dataloaders = [data_partitioner[i].get_distributed_data(batch_size=args.batch) for i in range(len(data_partitioner))]
        distributed_dataloaders = [
            distributed_dataloaders[i][j]
            for i in range(len(distributed_dataloaders))
            for j in range(data_partitioner[i].n_clients)
        ]
        print(len(distributed_dataloaders))
        fl_scen = FL_scenario(all_clients_weights, args.n_clients, args.n_sampled_clients, distributed_dataloaders)
    else:
        fl_scen = FL_scenario(
            data_partitioner.get_all_client_weights(),
            args.n_clients,
            args.n_sampled_clients,
            data_partitioner.get_distributed_data(batch_size=args.batch)
        )

    print(args.model_type)
    if args.model_type != 'L2P':
        raise ValueError('Model type is not supported')

    if args.query_model_type == 'vit_cls':
        server_model = to_device(
            L2P_ViT_B32(
                prompt_method=args.prompt_method,
                batchwise_prompt=args.batchwise_prompt,
                pool_size=args.pool_size,
                top_k=args.n_tokens,
                num_classes=num_classes,
                candidate_key=None
            ),
            args.device
        )
    elif args.query_model_type == 'cait_s24':
        server_model = to_device(
            L2P_CaiT_S24(
                prompt_method=args.prompt_method,
                batchwise_prompt=args.batchwise_prompt,
                pool_size=args.pool_size,
                top_k=args.n_tokens,
                num_classes=num_classes,
                candidate_key=None
            ),
            args.device
        )
    elif args.query_model_type == 'deit_b16':
        server_model = to_device(
            L2P_DeiT_B16(
                prompt_method=args.prompt_method,
                batchwise_prompt=args.batchwise_prompt,
                pool_size=args.pool_size,
                top_k=args.n_tokens,
                num_classes=num_classes,
                candidate_key=None,
                pretrained=True
            ),
            args.device
        )
    elif args.query_model_type == 'convit_base':
        server_model = to_device(
            L2P_ConViT_Base(
                prompt_method=args.prompt_method,
                batchwise_prompt=args.batchwise_prompt,
                pool_size=args.pool_size,
                top_k=args.n_tokens,
                num_classes=num_classes,
                candidate_key=None,
                pretrained=True
            ),
            args.device
        )
    else:
        raise ValueError('Model type is not supported')

    eval_pool_size = []
    eval_pool = []
    client_pool = []
    eval_pool_size.append(args.pool_size)
    eval_pool.append(server_model.pool.prompt.data.detach().cpu().numpy())
    server_model.build_trainable_keys()

    if '_' in args.alg:
        associate_algo = args.alg.split('_')[0]
        algclass = get_algorithm(associate_algo)
    else:
        associate_algo = args.alg
        algclass = get_algorithm(args.alg)

    if args.alg == 'fedavg_nonpara':
        if associate_algo != 'fedavg':
            raise ValueError('Algorithm is not supported')
        print('nonpara')
        algo = algclass(
            server_model=server_model,
            scenario=fl_scen,
            loss_fun=nn.CrossEntropyLoss(),
            fed_method='nonparametric_aggregation',
            nonpara_hidden=args.nonpara_hidden,
            class_mask=class_mask,
            device=args.device
        )
    elif args.alg == 'fedavg':
        print('normal aggregation')
        algo = algclass(
            server_model=server_model,
            scenario=fl_scen,
            loss_fun=nn.CrossEntropyLoss(),
            fed_method='aggregation',
            class_mask=class_mask,
            device=args.device
        )
    else:
        raise ValueError('Algorithm is not supported')

    if args.load_model is not None:
        load_model_trainable_part(algo.server_model, args.load_model)
        print(f'Model loaded from {args.load_model}')
        broadcast_like_communication(algo.server_model, algo.client_model, device=args.device)

    init_model_path = (
        f'./output_record/saved_model/saved_model_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}'
        f'clietns_{scenrio_type}_model{args.query_model_type}_{args.data_distribution}_alpha{args.alpha}_'
        f'{args.instance_label}_{defense_tag}_{inject_tag}_init.pt'
    )
    save_model_trainable_part(algo.server_model, init_model_path)
    print('\nSave init model\n')

    injection_batch = _resolve_dataset_sample(trainset, args.inject_sample_idx, args.inject_dataset_idx)
    injection_done = False

    for comm_round in trange(args.comms):
        if (
            not injection_done
            and args.inject_epoch > 0
            and (comm_round + 1) == args.inject_epoch
        ):
            injected_indices = inject_stealth_prompt_pool(algo.server_model, injection_batch, args)
            broadcast_like_communication(algo.server_model, algo.client_model, device=args.device)
            injection_done = True
            print(
                f'Stealth prompt-pool injection applied before round {comm_round + 1}; '
                f'updated slots: {injected_indices}'
            )

        algo.client_train(
            comm_round=comm_round,
            epochs=args.local_eps,
            lr=args.lr,
            dp_mechanism=args.dp_mechanism,
            eps=args.eps,
            frac=args.n_sampled_clients / args.n_clients,
            reduce_sim_scalar=args.reduce_sim_scalar,
            print_output=True,
            defense=args.defense,
            mixup_alpha=args.mixup_alpha,
            sparse_ratio=args.sparse_ratio,
            perturb_sigma=args.perturb_sigma,
            sampling_proportion=args.sampling_proportion,
            client_sampling_type=args.client_sampling_type
        )

        if args.model_type == 'L2P':
            for i in range(len(algo.selected_client_index)):
                client_pool.append(algo.client_model[i].pool.prompt.data.detach().cpu().numpy())

        algo.server_aggre()

        if args.model_type == 'L2P':
            eval_loss, eval_acc, current_pool_size = algo.server_eval(testloader, comm_round)
            eval_pool_size.append(current_pool_size)
            eval_pool.append(algo.server_model.pool.prompt.data.detach().cpu().numpy())
        else:
            eval_loss, eval_acc = algo.server_eval(testloader, comm_round)

        eval_loss_record.append(eval_loss)
        eval_acc_record.append(eval_acc)

    min_index = int(np.argmin(np.array(eval_loss_record)))
    print(f'Final accuracy with minimal eval_loss: {eval_acc_record[min_index]}')

    best_model_path = (
        f'./output_record/saved_model/saved_model_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}'
        f'clietns_{scenrio_type}_model{args.query_model_type}_{args.data_distribution}_alpha{args.alpha}_'
        f'{args.instance_label}_{defense_tag}_{inject_tag}_round_{args.comms}_acc_{eval_acc:.4f}.pt'
    )
    if args.save_model:
        save_model_trainable_part(algo.server_model, best_model_path)
        print(f'New last model saved to {best_model_path} with acc: {eval_acc:.4f}')