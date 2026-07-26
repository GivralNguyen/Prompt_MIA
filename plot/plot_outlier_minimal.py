import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from Scenario import FL_scenario
from util.TOdevice import to_device
from util.saving_tools import load_model_trainable_part
from Algo.get_algos import get_algorithm
from Models.L2P_heuristic_model import L2P_ViT_B32


class Args:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='Minimal outlier detection over a dataset subset.')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to use.')
    parser.add_argument('--model-path', default=None, help='Path to saved model trainable part.')
    parser.add_argument('--n-samples', type=int, default=20, help='Number of dataset samples to evaluate.')
    parser.add_argument('--k-top', type=int, default=4, help='Number of keys injected as adversarial outliers.')
    parser.add_argument('--pool-size', type=int, default=20, help='L2P key pool size.')
    parser.add_argument('--n-tokens', type=int, default=4, help='Number of query tokens used for similarity.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    return parser.parse_args()


def build_transform():
    norm_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    resize = 256
    centercrop_size = 224
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(centercrop_size),
        transforms.ToTensor(),
        transforms.Normalize(*norm_stats),
    ])


def generate_point_with_similarity(target_vec, similarity, benign_key):
    target_vec = target_vec.reshape(-1)
    benign_key = benign_key.reshape(-1)

    target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-12)
    benign_norm = benign_key / (np.linalg.norm(benign_key) + 1e-12)

    rand_vec = np.random.randn(benign_key.shape[0]).astype(np.float32)
    rand_vec /= (np.linalg.norm(rand_vec) + 1e-12)

    final_vec = rand_vec
    final_vec /= (np.linalg.norm(final_vec) + 1e-12)

    orthogonal_vec = final_vec - np.dot(final_vec, target_norm) * target_norm
    orthogonal_norm = orthogonal_vec / (np.linalg.norm(orthogonal_vec) + 1e-12)

    new_vec_norm = similarity * target_norm + np.sqrt(max(0.0, 1 - similarity**2)) * orthogonal_norm
    new_vec_norm /= (np.linalg.norm(new_vec_norm) + 1e-12)
    new_vec = new_vec_norm * (np.linalg.norm(target_vec) + 1e-12)
    return new_vec


def make_detectors():
    return {
        'IsolationForest': IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', random_state=42),
        'LocalOutlierFactor': LocalOutlierFactor(n_neighbors=5, contamination='auto', novelty=False),
        'OneClassSVM': OneClassSVM(kernel='rbf', gamma='scale', nu=0.2),
        'EllipticEnvelope': EllipticEnvelope(contamination=0.2, support_fraction=None, random_state=42, assume_centered=False),
    }


def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def build_algorithm(args, num_classes):
    server_model = to_device(
        L2P_ViT_B32(
            prompt_method='shallow',
            batchwise_prompt=False,
            pool_size=args.pool_size,
            top_k=args.n_tokens,
            num_classes=num_classes,
            candidate_key=None,
        ),
        args.device,
    )
    server_model.build_trainable_keys()

    scenario = FL_scenario(np.array([1]), n_clients=1, n_clients_each_round=1, distributed_dataloaders=[None])
    algo = get_algorithm('fedavg')(
        server_model=server_model,
        scenario=scenario,
        loss_fun=nn.CrossEntropyLoss(),
        class_mask=None,
        fed_method='aggregation',
        device=args.device,
    )
    return algo


def load_dataset(args):
    transform = build_transform()
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./Dataset', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./Dataset', train=False, download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./Dataset', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./Dataset', train=False, download=True, transform=transform)
        num_classes = 100
    else:
        raise ValueError('Unsupported dataset')
    return trainset, testset, num_classes


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainset, testset, num_classes = load_dataset(args)
    algo = build_algorithm(args, num_classes)

    if args.model_path is not None:
        load_model_trainable_part(algo.server_model, args.model_path)
        print(f'Loaded model trainable part from {args.model_path}')

    with torch.no_grad():
        fixed_keys = algo.server_model.pool.key.clone()
        fixed_key_norm_torch = algo.server_model.pool.l2_normalize(fixed_keys, dim=1)
        fixed_key_norm = fixed_key_norm_torch.cpu().numpy()

    sample_ids = np.random.choice(range(len(trainset)), size=min(args.n_samples, len(trainset)), replace=False)
    metrics = {
        name: {
            'precision': [],
            'recall': [],
            'f1': [],
            'tpr': [],
            'fpr': [],
        }
        for name in make_detectors().keys()
    }

    detectors = make_detectors()

    for sample_id in tqdm(sample_ids, desc='Processing samples'):
        x, label = trainset[sample_id]
        x_test, label_test = testset[sample_id]
        x = x.unsqueeze(0).to(args.device).float()
        x_test = x_test.unsqueeze(0).to(args.device).float()

        with torch.no_grad():
            feat_train = algo.server_model.get_cls_keys(x)
            x_embed_norm = algo.client_model[0].pool.l2_normalize(feat_train, dim=1)

        fixed_keys_np = fixed_keys.detach().cpu().numpy()
        fixed_key_norm_temp = fixed_key_norm.copy()
        similarity = torch.matmul(x_embed_norm, fixed_key_norm_torch.t())
        topk_vals, topk_idxs = torch.topk(similarity[0], k=min(args.k_top, fixed_key_norm.shape[0]), largest=True)
        chosen_idx = topk_idxs.cpu().numpy().tolist()

        mask = np.ones(similarity.shape[1], dtype=bool)
        mask[chosen_idx] = False
        non_adv_similarity = similarity[:, mask]
        max_val, _ = torch.max(non_adv_similarity, dim=1)
        similarity_start = float(torch.min(max_val + 0.02, torch.tensor(1.0, device=max_val.device)).cpu().item())
        similarity_end = float(torch.min(max_val + 0.07, torch.tensor(1.0, device=max_val.device)).cpu().item())
        varied_target_similarities = np.random.uniform(similarity_start, similarity_end, len(chosen_idx))

        non_chosen = np.where(mask)[0]
        anchor_idx = np.random.choice(non_chosen, size=len(chosen_idx), replace=False)
        similar_points = []
        query_vector_np = feat_train.detach().cpu().numpy().copy()
        for sim, idx in zip(varied_target_similarities, anchor_idx):
            benign_key = fixed_keys_np[idx:idx+1]
            similar_points.append(generate_point_with_similarity(query_vector_np[0], sim, benign_key))

        similar_points_tensor = torch.tensor(np.stack(similar_points, axis=0), dtype=fixed_keys.dtype, device=fixed_keys.device)
        similar_points_tensor_norm = algo.client_model[0].pool.l2_normalize(similar_points_tensor, dim=1)
        for i, idx in enumerate(chosen_idx):
            fixed_key_norm_temp[idx] = similar_points_tensor_norm[i].cpu().numpy()

        X = fixed_key_norm_temp
        X_scaled = StandardScaler().fit_transform(X)

        y_true = np.zeros(X.shape[0], dtype=int)
        y_true[np.array(chosen_idx, dtype=int)] = 1

        for name, model in detectors.items():
            if name == 'LocalOutlierFactor':
                labels = model.fit_predict(X_scaled)
                scores = -model.negative_outlier_factor_
            elif name == 'EllipticEnvelope':
                model.fit(X_scaled)
                dec = model.decision_function(X_scaled)
                labels = np.where(dec >= 0, 1, -1)
                scores = -dec
            else:
                model.fit(X_scaled)
                dec = model.decision_function(X_scaled)
                labels = np.where(dec >= 0, 1, -1)
                scores = -dec

            y_pred = (labels == -1).astype(int)
            TP = int(np.sum((y_pred == 1) & (y_true == 1)))
            FP = int(np.sum((y_pred == 1) & (y_true == 0)))
            FN = int(np.sum((y_pred == 0) & (y_true == 1)))
            TN = int(np.sum((y_pred == 0) & (y_true == 0)))

            precision = safe_div(TP, TP + FP)
            recall = safe_div(TP, TP + FN)
            f1 = safe_div(2 * precision * recall, precision + recall)
            tpr = recall
            fpr = safe_div(FP, FP + TN)

            metrics[name]['precision'].append(precision)
            metrics[name]['recall'].append(recall)
            metrics[name]['f1'].append(f1)
            metrics[name]['tpr'].append(tpr)
            metrics[name]['fpr'].append(fpr)

    print('\n=== Mean / Std over the sampled subset ===')
    for name, values in metrics.items():
        print(f'\n{name}:')
        for metric_name, values_list in values.items():
            mean_val = np.mean(values_list)
            std_val = np.std(values_list)
            print(f'  {metric_name}: {mean_val:.4f} ± {std_val:.4f}')


if __name__ == '__main__':
    main()
