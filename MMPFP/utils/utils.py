import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops
import time
import json
import random

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)

def count_parameters(model):
    return sum(p.size for p in model.get_parameters())

def accuracy(preds, labels):
    preds = ops.Argmax(axis=1)(preds)
    correct = (preds == labels).sum().asnumpy()
    total = labels.shape[0]
    return correct / total

def f1_score(preds, labels, num_classes=3):
    preds = ops.Argmax(axis=1)(preds)
    f1_scores = []
    for i in range(num_classes):
        tp = ((preds == i) & (labels == i)).sum().asnumpy()
        fp = ((preds == i) & (labels != i)).sum().asnumpy()
        fn = ((preds != i) & (labels == i)).sum().asnumpy()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    return np.mean(f1_scores)

def save_checkpoint(model, epoch, save_path="checkpoints"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ckpt_path = os.path.join(save_path, f"model_epoch_{epoch}.ckpt")
    ms.save_checkpoint(model, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

def load_checkpoint(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        ms.load_checkpoint(checkpoint_path, net=model)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")

def log_metrics(log_file, epoch, loss, acc, f1):
    log_data = {
        "epoch": epoch,
        "loss": loss,
        "accuracy": acc,
        "f1_score": f1
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_data) + "\n")

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏳ Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def normalize_tensor(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
