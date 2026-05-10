import torch
import numpy as np
import random
import time
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

def print_log(result_path, *args):
    os.makedirs(result_path, exist_ok=True)

    print(*args)
    file_path = result_path + '/log.txt'
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args, file=f)

def save_result(result_path, acc, mean_acc, confusion, detailed_confusion_matrix):
    # Assuming acc, mean_acc, confusion, and detailed_confusion_matrix are already defined
    # 1. Save Accuracy and Mean Accuracy as Text
    with open(result_path + "/accuracy.txt", "w") as f:
        f.write(f"Accuracy: {acc:.2f}\n")
        f.write(f"Mean Accuracy: {mean_acc:.2f}\n")
    
    # 2. Save Confusion Matrix as a Numpy Array
    np.save(result_path + "/confusion_matrix.npy", confusion)

    # 3. Save Detailed Confusion Matrix as a Dictionary
    with open(result_path + "/detailed_confusion_matrix.pkl", "wb") as f:
        pickle.dump(detailed_confusion_matrix, f)

def load_result(result_path):
    with open(result_path + "/accuracy.txt", "r") as f:
        lines = f.readlines()
        acc = float(lines[0].split(": ")[1])
        mean_acc = float(lines[1].split(": ")[1])
    
    confusion = np.load(result_path + "/confusion_matrix.npy")

    with open(result_path + "/detailed_confusion_matrix.pkl", "rb") as f:
        detailed_confusion_matrix = pickle.load(f)
    
    return acc, mean_acc, confusion, detailed_confusion_matrix

def map_detailed_confusion_matrix_to_file_names(test_set, detailed_confusion_matrix):
    transformed_matrix = {}
    for k, v in detailed_confusion_matrix.items():
        transformed_matrix[k] = [test_set.frames[i] for i in v]
    return transformed_matrix

def save_detailed_confusion_matrix_slot_by_slot(save_path, detailed_confusion_matrix):
    real_save_path = save_path + "/confusion_matrix_slots"
    os.makedirs(real_save_path, exist_ok=True)
    for k, v in detailed_confusion_matrix.items():
        with open(real_save_path + "/" + str(k[0] + 1) + str(k[1] + 1), 'w') as f:
            to_write = ""
            for filename_tuple in v:
                to_write += "'" + str(filename_tuple[0]) + "/" + str(filename_tuple[1]) + "',\n"
            f.write(to_write)
                
def convert_confusion_matrix_for_print(confusion_dict):
    # 获取所有出现过的预测标签和真实标签
    pred_labels = set()
    true_labels = set()
    
    for (pred, true) in confusion_dict.keys():
        pred_labels.add(pred)
        true_labels.add(true)
    
    # 合并所有唯一标签并排序（确保顺序一致）
    all_labels = sorted(pred_labels | true_labels)
    n = len(all_labels)
    
    # 创建标签到索引的映射
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}
    
    # 初始化全零的二维矩阵
    matrix = [[0] * n for _ in range(n)]
    
    # 填充矩阵
    for (pred, true), samples in confusion_dict.items():
        if pred in label_to_index and true in label_to_index:
            i = label_to_index[pred]
            j = label_to_index[true]
            matrix[i][j] = len(samples)
    
    return matrix, all_labels


# ------------------------------------------------------------------------
# Reference:
# https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py
# ------------------------------------------------------------------------
import inspect
class HyperParameters:  
    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


def append_text_to_file(save_path, filename, *args):
    try:
        full_path = os.path.join(save_path, filename)

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        with open(full_path, 'a', encoding='utf-8') as file:
            for content in args:
                if not isinstance(content, str):
                    content = str(content)
                file.write(content + '\n')  # 每个内容后添加换行符
        
        return True
    
    except Exception as e:
        print(f"Error when appending text to a file: {str(e)}")
        return False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = val.detach().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """
    class to do timekeeping
    """
    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time
    

def set_random_seeds(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def steal_only_a_little_gpu(dataset, size=10):
    import torchvision.transforms as transforms
    tl = list(dataset.transform.transforms)
    tl[0] = transforms.Resize((size, size))  # The first one was Resize
    dataset.transform = transforms.Compose(tl)

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool type must be of forms True/False or yes/no')

def get_activation(activation, functional=True):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu if functional else nn.ReLU
    if activation == "gelu":
        return F.gelu if functional else nn.GELU
    if activation == "glu":
        return F.glu  if functional else nn.GLU
    if activation == "silu":
        return F.silu if functional else nn.SiLU
    if activation == "sigmoid":
        return F.sigmoid if functional else nn.Sigmoid
    raise RuntimeError(F"activation should not be {activation}.")