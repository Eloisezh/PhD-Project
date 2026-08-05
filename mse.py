#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import json
import torch
import numpy as np
from tqdm import tqdm


def _quantize_and_mse(W, num_bits=2):
	q_max = (2 ** (num_bits - 1)) - 1
	max_val = torch.max(torch.abs(W))
	if max_val == 0:
		return 0.0
	scale = max_val / (q_max + 1)
	W_quant_int = torch.round(W / scale)
	W_quant_int = torch.clamp(W_quant_int, -q_max, q_max)
	Q_W = W_quant_int * scale
	sensitivity = torch.sum((W - Q_W) ** 2).item()
	return sensitivity


# In[22]:


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


# In[23]:


from unittest import result


class mse():
	def __init__(self, model_path="yolo26n-seg.onnx", target=0.35):
			self.model_path = model_path
			self.target = target

	def filter(self):
		num_bits=8
		layer_scores = []
		layer_names = []
		weight_params = {}
		import onnx
		from onnx import numpy_helper
		import numpy as np
		import cv2
		import numpy as np
		from matplotlib import pyplot as plt

		model = onnx.load(self.model_path)
		inits = {init.name: init for init in model.graph.initializer}  # Get the initializers of the original modelq
		for init in model.graph.initializer:
			if "weight" in init.name and "conv" in init.name:
				# print(init.name)
				scores = []


				layer_names.append(init.name)
				total_params = 0
				W_np = numpy_helper.to_array(inits[init.name])
				weight_matrices = torch.from_numpy(W_np).float()
				w_flat = weight_matrices.view(-1).float()
				size_w = w_flat.numel()
				weight_params[init.name] = size_w
				# print(weight_matrices.shape)

				if weight_matrices.numel() == 0:
					layer_scores.append(0.0)
					continue

				total_mse = sum(_quantize_and_mse(W, num_bits) for W in weight_matrices)
				layer_score = total_mse / len(weight_matrices)
				layer_scores.append(layer_score)

		print("weight_params:", weight_params)
		output_dir = "./results"
		model_name = os.path.basename(self.model_path.rstrip("/"))
		os.makedirs(output_dir, exist_ok=True)
		out_file = os.path.join(output_dir, f"{model_name}_mse.json")
		result = {"scores": layer_scores, "layer_names": layer_names}
		with open(out_file, "w") as f:
			json.dump(result, f, indent=2)

		raw_scores = result.get("scores", [])
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
		print(len(selected))
		print(acc)
		print(selected)
		return selected


# In[24]:


if __name__ == '__main__':
	mse = mse()
	mse.filter()


# In[25]:


# inits = {init.name: init for init in model.graph.initializer}
# init_to_node = {}
# for node in model.graph.node:
# 	for inp in node.input:
# 		if inp in inits:
# 			init_to_node[inp] = node.name

# print(init_to_node.keys())
# # 使用
# print(init_to_node)


# In[26]:


# target_ratio = 0.35
# initializers = {init.name: init for init in model.graph.initializer}
# quantizable_ops = {"Conv"}

# node_weights = []

# for node_name in ranked_nodes:
# 	for node in model.graph.node:
# 		if node.name == init_to_node[f"{node_name}.weight"]:
# 			params = sum(
# 				int(np.prod(initializers[inp].dims))
# 				for inp in node.input if inp in initializers
# 			)
# 			if params > 0:
# 				node_weights.append((node.name, params))
# 			break

# total = sum(w for _, w in node_weights)
# target = (1 - target_ratio) / 0.75 * total

# # rng = np.random.default_rng(seed)
# # indices = rng.permutation(len(node_weights))

# selected, acc = [], 0
# for i in range(len(node_weights)):
# 	name, w = node_weights[i]
# 	selected.append(name)
# 	acc += w
# 	if acc >= target:
# 		break

# print(acc)
# print(target)
# print(len(ranked_nodes))
# print(len(selected))
# selected

