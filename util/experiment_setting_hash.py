import argparse
import torch
import torch.nn as nn
import hashlib
import json
import copy

def namespace_hash(ns: argparse.Namespace, ignore_in_ns: list=[]) -> str:
    """对 Namespace 做稳定哈希"""
    d = copy.deepcopy(vars(ns))
    # 忽略device
    for key_name in ignore_in_ns:
        if key_name in d:
            del d[key_name]
    # 排序保证稳定性
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def module_hash(model: nn.Module, include_weights: bool = True) -> str:
    """对模型做哈希"""
    h = hashlib.sha256()
    # 模型结构
    h.update(str(model).encode("utf-8"))

    if include_weights:
        # 把权重参数也算进去
        for k, v in model.state_dict().items():
            h.update(k.encode("utf-8"))
            h.update(v.cpu().numpy().tobytes())

    return h.hexdigest()

def combined_hash(ns: argparse.Namespace, model: nn.Module, include_weights=True, ignore_in_ns: list=[]) -> str:
    """综合哈希"""
    h = hashlib.sha256()
    h.update(namespace_hash(ns, ignore_in_ns).encode("utf-8"))
    h.update(module_hash(model, include_weights).encode("utf-8"))
    return h.hexdigest()


# ---- 示例 ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args([])

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    print("Namespace hash:", namespace_hash(args))
    print("Model hash (struct):", module_hash(model, include_weights=False))
    print("Model hash (with weights):", module_hash(model, include_weights=True))
    print("Combined hash:", combined_hash(args, model))
