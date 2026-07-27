# PhD-Project
- root
  - calibration_data
  - q_models
  - INT8.ipynb
  - INT8.py
  - INT16.ipynb
  - INT16.py
  - Qall.ipynb

How to run?
```
jupyter nbconvert --to script INT8.ipynb
jupyter nbconvert --to script INT16.ipynb
```
## 环境依赖

```bash
pip install -r requirements.txt
pip install pyyaml
```

## 使用方法

### 1. 跑单个量化任务

```bash
python run_quant.py --config configs/run.yaml
```

### 2. 批量跑多个量化任务

```bash
python run_quant.py --config configs/batch.yaml
```

### 3. 命令行临时覆盖参数（仅单任务配置生效）

```bash
python run_quant.py --config configs/example.yaml --tag test-debug --model yolo26s-seg.onnx
```

| 参数 | 是否必填 | 说明 |
|---|---|---|
| `--config` / `-c` | 是 | 配置文件路径 |
| `--model` | 否 | 覆盖配置文件中的 model |
| `--data` | 否 | 覆盖配置文件中的 data |
| `--tag` | 否 | 覆盖配置文件中的 tag |

## 配置文件格式

### 单任务格式（`configs/run.yaml`）

```yaml
model: "yolo26n-seg.onnx"
data: "coco.yaml"
tag: "our-35"
print_original: false
print_quantized: true
nodes:
  - '/model.2/m.0/cv1/conv/Conv'
  - '/model.2/cv1/conv/Conv'
```

| 字段 | 是否必填 | 说明 |
|---|---|---|
| `model` | 是 | 待量化的 ONNX 模型路径 |
| `data` | 是 | 校准数据集配置文件（如 coco.yaml） |
| `tag` | 是 | 本次任务标识，用于输出文件命名 |
| `nodes` | 是 | 需要量化处理的节点名称列表 |
| `print_original` | 否（默认 false） | 是否打印原始模型信息 |
| `print_quantized` | 否（默认 true） | 是否打印量化后模型信息 |

### 批量任务格式（`configs/batch.yaml`）

```yaml
model: "yolo26n-seg.onnx"    # 全局默认值
data: "coco.yaml"
print_original: false
print_quantized: true

tasks:
  - tag: "ewq-35"
    model: "yolo26n-seg-ewq.onnx"   # 可单独覆盖全局默认值
    nodes:
      - '/model.2/m.0/cv1/conv/Conv'

  - tag: "kurtboost-35"
    nodes:
      - '/model.10/m/m.0/attn/pe/conv/Conv'
```

规则：
- 顶层的 `model` / `data` / `print_original` / `print_quantized` 是所有任务的**默认值**
- 每个 `tasks` 条目可单独覆盖 `model`、`data` 等任意字段
- 每个任务**必须**自带 `tag` 和 `nodes`

## 输出示例
