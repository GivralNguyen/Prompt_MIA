import numpy as np
import torch
from collections import OrderedDict

def save_eval_npy_file(eval_loss_record, eval_acc_record, eval_pool_size=None, file_path='./'):
    if eval_pool_size is None:
        d = {'eval_loss': eval_loss_record,
             'eval_acc': eval_acc_record}
    else:
        d = {'eval_loss': eval_loss_record,
             'eval_acc': eval_acc_record,
             'pool_size': eval_pool_size}
    np.save(file_path, d)
    
def save_model_trainable_part(model, file_path):
    model.eval()
    if len(model.trainable_keys) == 0:
        raise ValueError("No trainable part should be loaded or Miss building trainable_keys")
    else:
        trainable_part_dict = OrderedDict()
        for key in model.trainable_keys:
            trainable_part_dict[key] = model.state_dict()[key]
        torch.save(trainable_part_dict, file_path)

def load_model_trainable_part(model, file_path):
    load_dict = torch.load(file_path)
    model_dict = model.state_dict()

    # check which keys exist in both
    model_keys = set(model.trainable_keys)
    ckpt_keys = set(load_dict.keys())
    common_keys  = model_keys & ckpt_keys
    # if len(load_dict) != len(model.trainable_keys):
    #     raise ValueError("Keys don't match")
    # else:
    #     for key in model.trainable_keys:
    #         model.state_dict()[key].copy_(load_dict[key])
    for key in common_keys:
        if load_dict[key].shape == model_dict[key].shape:
            model_dict[key].copy_(load_dict[key])
        else:
            print(f"⚠️ Shape mismatch for key '{key}': "
                  f"expected {tuple(model_dict[key].shape)}, got {tuple(load_dict[key].shape)}")