#!/usr/bin/env python
# coding: utf-8

# In[21]:


import onnx
import pandas as pd
import re
from onnx import numpy_helper
import numpy as np
from pathlib import Path
import json
from pathlib import Path


# In[22]:


def find_closest_below(d, target):
    # 第一步:筛选出所有小于target的键值对
    candidates = {k: v for k, v in d.items() if v < target}
    # print(type(candidates))

    if not candidates:
        return [], None  # 没有任何value小于target

    # 第二步:找出候选里最大的value(最接近target但仍小于target)
    # print(type(candidates.values()))
    best_value = max(candidates.values())

    # 第三步:找出所有value等于best_value的key(可能不止一个)
    best_keys = [k for k, v in candidates.items() if v == best_value]

    return best_keys, best_value


# In[ ]:


class statisticFilter():
    def __init__(
        self,
        model_path="yolo26n-seg.onnx",
        excel_path="onnx_weight_stats-26.xlsx",
        b_idx=10,
        n_idx=22,
        target=0.35,
        cache_path="alpha_beta_cache_mean_only.json"
    ):
        self.model_path = model_path
        self.excel_path = excel_path
        self.b_idx = b_idx
        self.n_idx = n_idx
        self.target = target
        self.cache_path = Path(cache_path)

    def filter(self):
        model = onnx.load(self.model_path)

        total_param = sum(
            int(np.prod(init.dims))
            for init in model.graph.initializer
        )

        print(f"total_param: {total_param}")

        target_param = (1 - self.target) / 0.75 * total_param

        # 使用完整路径区分不同模型
        model_key = str(Path(self.model_path).resolve())

        # 读取全部缓存
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        else:
            cache = {}

        # 当前模型已有缓存，直接使用
        if model_key in cache:
            print("Loading alpha-beta results from cache")

            alpha_beta_params = cache[model_key]["params"]
            alpha_beta_nodes = cache[model_key]["nodes"]

        # 当前模型不存在，重新计算
        else:
            print("Model not found in cache, calculating...")

            xl = pd.ExcelFile(self.excel_path)
            sections = xl.sheet_names

            section_statistics = {}
            alpha_beta_params = {}
            alpha_beta_nodes = {}
            alpha_beta_bounds = {}
            weight_params = {}

            ### 1. calculate per-section statistics
            for section in sections:
                df = xl.parse(section)

                section_statistics[section] = {
                    "mm": float(df["mean"].mean()),
                    "Astd": float(df["mean"].std()),
                    "Bstd": float(df["std"].mean()),
                }

            #### helper func
            inits = {
                init.name: init
                for init in model.graph.initializer
                if "weight" in init.name and "conv" in init.name
            }

            init_to_node = {}

            for node in model.graph.node:
                init_to_node.update({
                    inp: node.name
                    for inp in node.input
                    if inp in inits
                })

                n_params = sum(
                    int(np.prod(inits[inp].dims))
                    for inp in node.input
                    if inp in inits
                )

                if n_params > 0:
                    weight_params[node.name] = n_params
            #### end helper func

            #### 2. calculate params each alpha-beta pair
            for ta_int in range(1, 51):
                ta = ta_int / 10

                for tb in range(1, 51):
                    key = f"{ta:.1f}-{tb}"

                    alpha_beta_params[key] = 0
                    alpha_beta_nodes[key] = []

                    ### 1. calculate bounds
                    for section in sections:
                        mm = section_statistics[section]["mm"]
                        Astd = section_statistics[section]["Astd"]
                        Bstd = section_statistics[section]["Bstd"]

                        alpha_beta_bounds[section] = {
                            "lower_m": mm - ta * Astd,
                            "upper_m": mm + ta * Astd,
                            "lower_minmax": mm - tb * Bstd,
                            "upper_minmax": mm + tb * Bstd,
                        }

                    ### 2. calculate params
                    for initializer in model.graph.initializer:
                        if (
                            "conv" not in initializer.name
                            or "weight" not in initializer.name
                        ):
                            continue

                        w = numpy_helper.to_array(initializer)

                        mean = float(w.mean())
                        max_ = float(w.max())
                        min_ = float(w.min())

                        m = re.search(r"model\.(\d+)", initializer.name)

                        if m is None:
                            continue

                        idx = int(m.group(1))

                        section = (
                            "backbone" if idx < self.b_idx
                            else "neck" if idx < self.n_idx
                            else "head"
                        )

                        b = alpha_beta_bounds[section]

                        mean_ok = (
                            b["lower_m"]
                            <= mean
                            <= b["upper_m"]
                        )

                        range_ok = (
                            b["lower_minmax"] <= min_
                            and max_ <= b["upper_minmax"]
                        )

                        # if mean_ok and range_ok:
                        if mean_ok:
                            node_name = init_to_node[initializer.name]
                            param = weight_params[node_name]

                            alpha_beta_params[key] += int(param)
                            alpha_beta_nodes[key].append(node_name)

            # 保存当前模型结果，不影响其他模型
            cache[model_key] = {
                "params": alpha_beta_params,
                "nodes": alpha_beta_nodes,
            }

            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

            print(f"Cache saved to: {self.cache_path}")

        keys, value = find_closest_below(
            alpha_beta_params,
            target_param
        )

        if keys is None or len(keys) == 0:
            print("No alpha-beta pair found below target_param.")
            print(f"target_param: {target_param}, closest value: {value}")
            return None
        selected_key = keys[0]
        selected = alpha_beta_nodes[selected_key]

        print("alpha-beta:", selected_key)
        print("target params:", target_param)
        print("selected params:", alpha_beta_params[selected_key])
        print("selected nodes:", len(selected))

        return selected


# In[24]:


if __name__ == '__main__':
    filter = statisticFilter()
    filter.statistic_filter()

