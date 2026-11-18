import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file as save_safetensors


def convert_to_json_serializable(obj):
    """递归转换所有值为JSON可序列化类型（处理numpy.float32等）"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def loss_func(output, target, binary=False):
    if binary:
        return F.binary_cross_entropy(output, target)
    return torch.mean(torch.abs(output - target))


def evaluate(model, generator, cuda, move_data_to_gpu=None, inverse_transform_fn=None):
    """
    Evaluate model on data produced by `generator`.

    Args:
        model: PyTorch model
        generator: yields (batch_x, batch_y)
        cuda: bool
        move_data_to_gpu: optional function(move, cuda) -> Variable/Tensor. If None, imported from `models`.
        inverse_transform_fn: optional function to inverse-transform outputs/targets. If None, will try
            generator.inverse_transform_y, generator.inverse_transform, or fallback to identity.
    """
    # lazy import defaults
    if move_data_to_gpu is None:
        try:
            from models import move_data_to_gpu as _mdtg
            move_data_to_gpu = _mdtg
        except Exception:
            # fallback to identity that wraps arrays to tensors
            def move_data_to_gpu(x, cuda):
                import torch
                t = torch.tensor(x, dtype=torch.float32)
                if cuda:
                    t = t.cuda()
                from torch.autograd import Variable
                return Variable(t)

    if inverse_transform_fn is None:
        if hasattr(generator, 'inverse_transform_y'):
            inverse_transform_fn = getattr(generator, 'inverse_transform_y')
        elif hasattr(generator, 'inverse_transform'):
            inverse_transform_fn = getattr(generator, 'inverse_transform')
        else:
            inverse_transform_fn = lambda x: x

    model.eval()
    outputs_list, targets_list = [], []
    with torch.no_grad():
        for batch_x, batch_y in generator:
            batch_x = move_data_to_gpu(batch_x, cuda)
            output = model(batch_x)
            outputs_list.append(output.data.cpu().numpy())
            targets_list.append(batch_y)

    if len(outputs_list) == 0:
        return {'MAE': float('nan')}

    outputs = np.concatenate(outputs_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    outputs = inverse_transform_fn(outputs)
    targets = inverse_transform_fn(targets)

    mae = np.mean(np.abs(outputs - targets))
    return {'MAE': mae}


def save_best_model(model, model_dir, config):
    create_path = os.path.dirname(model_dir)
    if not os.path.exists(create_path):
        os.makedirs(create_path)
    save_safetensors(model.state_dict(), model_dir)


def save_history(history, log_dir):
    history_json = convert_to_json_serializable(history)
    with open(os.path.join(log_dir, 'training_history.json'), 'w') as f:
        json.dump(history_json, f)
