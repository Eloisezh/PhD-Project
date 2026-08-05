#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import json
import numpy as np
from scipy.stats import kurtosis
from tqdm import tqdm
import torch
import onnx
from onnx import numpy_helper
import numpy as np
import cv2
import numpy as np
from matplotlib import pyplot as plt


def _matrix_kurtosis(weight_matrix):
    w = weight_matrix.detach().cpu().float().numpy().flatten()
    return float(kurtosis(w, axis=None, fisher=True, bias=True, nan_policy="omit"))

def _compute_difference_sequence(scores):
    diffs = [scores[l + 1] - scores[l] for l in range(len(scores) - 1)]
    if not diffs:
        return [0.0], [False]
    mu = float(np.mean(diffs))
    sigma = float(np.std(diffs)) + 1e-12
    z_diffs = [abs(d - mu) / sigma for d in diffs]
    outliers = [z > 2 for z in z_diffs]
    outliers = [False] + outliers
    return z_diffs, outliers


# In[18]:


def _normalize_scores(scores):
    arr = np.array(scores, dtype=np.float64)
    min_val = arr.min()
    max_val = arr.max()
    normalized = (arr - min_val) / (max_val + 1e-8)
    return normalized.tolist()


def _rank_layers(scores):
    arr = np.array(scores)
    ranked = np.argsort(arr).tolist()
    return ranked


def _build_output(metric_name, model_path, scores, extra=None):
    model_name = os.path.basename(model_path.rstrip("/\\"))
    num_layers = len(scores)
    layer_labels = [f"layer_{i}" for i in range(1, num_layers + 1)]
    layer_map = {label: float(s) for label, s in zip(layer_labels, scores)}
    out = {
        "metric": metric_name,
        "model": model_name,
        "num_layers": num_layers,
        "layer_scores": layer_map,
        "ranked_layers": _rank_layers(scores),
        "scores": scores,
    }
    if extra:
        out.update(extra)
    return out

def _get_sensitive_nodes():
    nodes = []

    return nodes

def _save_output(output_dict, output_dir, metric_name, model_path):
    model_name = os.path.basename(model_path.rstrip("/\\"))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_{metric_name}.json")
    with open(out_path, "w") as f:
        json.dump(output_dict, f, indent=2)
    return out_path


# In[19]:


class kurtboost():
    def __init__(self, model_path="yolo26n-seg.onnx", target=0.35):
        self.model_path = model_path
        self.target = target
    def filter(self):
        threshold = 1.0
        raw_scores = []
        layer_scores = []
        layer_names = []
        weight_params = {}


        model = onnx.load(self.model_path)
        inits = {init.name: init for init in model.graph.initializer}  # Get the initializers of the original modelq
        for init in model.graph.initializer:
            if "weight" in init.name and "conv" in init.name:
                # print(init.name)

                layer_names.append(init.name)
                total_params = 0
                W_np = numpy_helper.to_array(inits[init.name]).flatten()

                weights = torch.from_numpy(W_np).float()
                w_flat = weights.view(-1).float()
                size_w = w_flat.numel()
                weight_params[init.name] = size_w

                kurt_vals = float(kurtosis(W_np, axis=None, fisher=True, bias=True, nan_policy="omit"))
                # kurt_vals = torch.from_numpy(W_np).float().flatten()
                raw_scores.append(float(np.sum(kurt_vals)) if kurt_vals else 0.0)
        print(len(layer_names))	
        _, outlier_flags = _compute_difference_sequence(raw_scores)

        l_idx = 0
        for init in model.graph.initializer:
            if "weight" in init.name and "conv" in init.name:
                print(init.name)

                W_np = numpy_helper.to_array(inits[init.name])
                kurt_vals = torch.from_numpy(W_np).float()
                boost = 2.0 if l_idx < len(outlier_flags) and outlier_flags[l_idx] else 1.0
                layer_scores.append(raw_scores[l_idx] * boost)
                l_idx += 1

        output_dir = "./results"
        model_name = os.path.basename(self.model_path.rstrip("/"))
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"{model_name}_kurtboost.json")
        result = {"scores": layer_scores, "layer_names": layer_names}
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        vraw_scores = result.get("scores", [])
        layer_names = result.get("layer_names", [])
        print(raw_scores)
        print(layer_names)
        raw_scores = _normalize_scores(raw_scores)

        # extra_keys = {k: v for k, v in result.items() if k != "scores"}
        # output_dict = _build_output(args.metric, args.model_path, raw_scores, extra=extra_keys)

        # out_path = _save_output(output_dict, args.output_dir, args.metric, args.model_path)
        # print(f"[+] Results saved to: {out_path}")
        nodes_to_quantize = []
        ranked_nodes = []
        ranked_weights = []

        print(f"\n[*] Top-5 most sensitive layers (0-indexed):")
        ranked = _rank_layers(raw_scores)
        for rank, layer_idx in enumerate(ranked[:5]):
            print(f"    Rank {rank + 1}: Layer {layer_names[layer_idx].replace('.weight', '')}  (score={raw_scores[layer_idx]:.6f})")
            nodes_to_quantize.append(layer_names[layer_idx].replace('.weight', ''))
        nodes_to_quantize

        for rank, layer_idx in enumerate(ranked):
            ranked_nodes.append(layer_names[layer_idx].replace('.weight', ''))
            ranked_weights.append(layer_names[layer_idx])

        ranked_nodes


        total_param = sum(int(np.prod(init.dims)) for init in model.graph.initializer)
        target_param = (1 - self.target) / 0.75 * total_param

        inits = {init.name: init for init in model.graph.initializer if "weight" in init.name and "conv" in init.name}
        init_to_node = {}
        for node in model.graph.node:
            ### init_to_node ref
            init_to_node.update({inp: node.name for inp in node.input if inp in inits})
        print(init_to_node)

        print(weight_params)
        selected, acc = [], 0
        for weight in ranked_weights:
            w = weight_params[weight]
            selected.append(init_to_node[weight])
            acc += w
            if acc >= target_param:
                acc -= w
                break

        # print(acc)
        # print(target_param)
        # print(len(ranked_nodes))
        # print(len(selected))
        selected
        print("len of selected:",len(selected))
        print("num of selected params:", acc)
        print("selected nodes:", selected)
        return selected


# In[20]:


if __name__ == '__main__':
	kurtboost = kurtboost(model_path="yolo26n-seg.onnx", target=0.35)
	kurtboost.filter()


# In[21]:


# import re

# def normalize(name: str) -> str:
# 	# 去掉 ONNX 命名里末尾的算子类型（如 /Conv, /Gemm 等大写开头的算子名）
# 	name = re.sub(r'/[A-Z][A-Za-z0-9_]*$', '', name)
# 	# 把 / 换成 .
# 	name = name.replace('/', '.')
# 	# 去掉开头可能多出来的 . （比如原本是 /model.10/... 转换后变成 .model.10....）
# 	name = name.strip('.')
# 	# 去掉像 m.m.0 这种重复段（ONNX 常见的 "父模块名.父模块名.子索引" 冗余）
# 	parts = name.split('.')
# 	cleaned = []
# 	for i, p in enumerate(parts):
# 		if i > 0 and cleaned and p == cleaned[-1]:
# 			continue
# 		cleaned.append(p)
# 	return '.'.join(cleaned)


# In[ ]:


# inits = {init.name: init for init in model.graph.initializer}
# init_to_node = {}
# for node in model.graph.node:
#     for inp in node.input:
#         if inp in inits:
#             init_to_node[inp] = node.name

# print(init_to_node.keys())
# # 使用
# print(init_to_node)


# In[ ]:


# target_ratio = 0.35
# initializers = {init.name: init for init in model.graph.initializer}
# quantizable_ops = {"Conv"}

# node_weights = []
# i = 0
# for node_name in ranked_nodes:
#     for node in model.graph.node:
#         if node.name == init_to_node[f"{node_name}.weight"]:
#             params = sum(
#                 int(np.prod(initializers[inp].dims))
#                 for inp in node.input if inp in initializers
#             )
#             if params > 0:
#                 node_weights.append((node.name, params))
#             break
# print("len",len(node_weights))
# total = sum(w for _, w in node_weights)
# target = (1 - target_ratio) / 0.75 * total

# # rng = np.random.default_rng(seed)
# # indices = rng.permutation(len(node_weights))

# selected, acc = [], 0
# for i in range(len(node_weights)):
#     name, w = node_weights[i]
#     selected.append(name)
#     acc += w
#     if acc >= target:
#         break
# print(acc)
# print(target)

# print(len(ranked_nodes))
# print(len(selected))
# selected

