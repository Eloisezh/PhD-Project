#!/usr/bin/env python3
"""
QAL 静态量化批量运行脚本

用法:
    单任务:  python run_quant.py --config configs/example.yaml
    多任务:  python run_quant.py --config configs/batch_example.yaml
    覆盖参数: python run_quant.py --config configs/example.yaml --model other.onnx --tag test-run
"""

import argparse
import sys
from pathlib import Path

import yaml

import INT8


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"配置文件为空或格式错误: {config_path}")
    return cfg


def build_task_list(cfg: dict) -> list[dict]:
    """
    把配置统一展开成任务列表。
    支持两种格式:
      1. 单任务: 顶层直接有 model/data/tag/nodes
      2. 多任务: 顶层有 tasks: [...]，全局字段作为默认值，可被单个 task 覆盖
    """
    if "tasks" in cfg:
        global_defaults = {k: v for k, v in cfg.items() if k != "tasks"}
        tasks = []
        for task in cfg["tasks"]:
            merged = {**global_defaults, **task}
            tasks.append(merged)
        return tasks
    else:
        return [cfg]


def validate_task(task: dict, index: int):
    required = ["model", "data", "tag", "nodes"]
    missing = [k for k in required if k not in task or task[k] in (None, "")]
    if missing:
        raise ValueError(f"第 {index} 个任务缺少必填字段: {missing}\n任务内容: {task}")
    if not isinstance(task["nodes"], list) or len(task["nodes"]) == 0:
        raise ValueError(f"第 {index} 个任务的 nodes 必须是非空列表\n任务内容: {task}")


def run_task(task: dict, index: int, total: int):
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] 开始量化任务: tag={task['tag']}")
    print(f"  model = {task['model']}")
    print(f"  data  = {task['data']}")
    print(f"  nodes = {len(task['nodes'])} 个节点")
    print(f"{'='*60}")

    quantizer = INT8.StaticQuantizer(
        task["model"],
        task["data"],
        task["nodes"],
        task["tag"],
        print_original=task.get("print_original", False),
        print_quantized=task.get("print_quantized", True),
    )
    q_model = str(quantizer.static_quantize())

    print(f"[{index}/{total}] 完成: tag={task['tag']} -> {q_model}")
    return q_model


def parse_args():
    parser = argparse.ArgumentParser(description="QAL 静态量化批量运行工具")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="配置文件路径 (yaml)，支持单任务或多任务(tasks)格式"
    )
    # 命令行覆盖参数，仅在单任务配置下生效，方便临时改一两个字段而不改文件
    parser.add_argument("--model", default=None, help="覆盖配置文件中的 model")
    parser.add_argument("--data", default=None, help="覆盖配置文件中的 data")
    parser.add_argument("--tag", default=None, help="覆盖配置文件中的 tag")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    tasks = build_task_list(cfg)

    # 命令行覆盖，只在单任务且用户显式传了对应参数时生效
    if len(tasks) == 1:
        if args.model:
            tasks[0]["model"] = args.model
        if args.data:
            tasks[0]["data"] = args.data
        if args.tag:
            tasks[0]["tag"] = args.tag

    for i, task in enumerate(tasks, start=1):
        validate_task(task, i)

    results = []
    for i, task in enumerate(tasks, start=1):
        try:
            q_model = run_task(task, i, len(tasks))
            results.append((task["tag"], q_model, "success"))
        except Exception as e:
            print(f"[错误] tag={task['tag']} 量化失败: {e}", file=sys.stderr)
            results.append((task["tag"], None, f"failed: {e}"))

    print(f"\n{'='*60}")
    print("批量任务汇总:")
    for tag, path, status in results:
        print(f"  - {tag}: {status}" + (f" -> {path}" if path else ""))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()