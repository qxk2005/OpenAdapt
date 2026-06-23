# OpenAdapt GUI 自动化 — 录制、训练与回放

基于 [OpenAdapt](https://github.com/OpenAdaptAI/OpenAdapt) 录制 GUI 操作，使用 VL（Vision-Language）模型进行微调训练，实现操作的 **1:1 精确回放** 或 **模型推理回放**。

> **环境**: macOS (Apple Silicon) · Python 3.11+ · pyenv 虚拟环境 `OAT`

---

## 📁 项目结构

```
.
├── my-task/                    # 录制数据目录
│   ├── recording.db            # 原始操作事件数据库（174 个事件）
│   ├── screenshots/            # 每一步的截图（12 张）
│   └── oa_recording-*.mp4      # 屏幕录像
├── train_local.py              # 🎯 本地训练脚本（绕过 CLI 兼容性问题）
├── train_config.yaml           # 训练超参数配置
├── training_output/            # 训练产出
│   ├── checkpoint-*/           # 中间检查点（每 epoch 保存）
│   └── final/                  # 最终模型（LoRA adapter）
├── run_model.py                # 🤖 基于训练模型推理回放
├── replay_capture.py           # 🔄 1:1 精确回放录制操作
├── extract_demo.py             # 📝 提取操作轨迹为 demo.txt
└── migrate_recording.py        # 🔧 数据库格式迁移工具
```

---

## 🎬 完整工作流

### 1. 录制 GUI 操作

使用 OpenAdapt CLI 录制屏幕操作：

```bash
# 开始录制（录制名称为 my-task）
openadapt capture start --name my-task

# 执行你要录制的操作...

# 停止录制
openadapt capture stop
```

录制产出保存在 `./my-task/` 目录下。

---

### 2. 数据迁移（可选）

如果需要将旧格式 `recording.db` 迁移为新格式 `capture.db`（供 openadapt-ml 训练管线使用）：

```bash
# 预览迁移内容（不修改文件）
python migrate_recording.py --capture ./my-task --dry-run

# 执行迁移
python migrate_recording.py --capture ./my-task
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--capture, -c` | `./my-task` | 录制数据目录路径 |
| `--dry-run, -n` | — | 仅预览，不实际创建文件 |

---

### 3. 提取 Demo 轨迹文本

从录制中提取人类可读的操作步骤描述：

```bash
# 默认提取到 ./my-task/demo.txt
python extract_demo.py

# 指定路径
python extract_demo.py --capture ./my-task --output ./demo.txt

# 覆盖任务描述
python extract_demo.py --task "打开计算器计算 1+1"
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--capture, -c` | `./my-task` | 录制数据目录路径 |
| `--output, -o` | `<capture>/demo.txt` | 输出文件路径 |
| `--task, -t` | 从 DB 读取 | 覆盖任务描述 |

提取结果可用于 OpenAdapt eval：
```bash
openadapt eval run --agent api-openai --demo ./my-task/demo.txt
```

---

### 4. 训练 VL 模型

使用 `train_local.py` 对 VL 模型进行 LoRA 微调：

```bash
# 默认训练（Qwen3-VL-2B, 40 epochs）
python train_local.py

# 指定模型和轮数
python train_local.py --model Qwen/Qwen2.5-VL-3B-Instruct --epochs 20

# 预览训练数据（不实际训练）
python train_local.py --dry-run

# 完整参数示例
python train_local.py \
    --capture ./my-task \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --output training_output \
    --epochs 40 \
    --lr 2e-4 \
    --batch-size 1 \
    --lora-r 32
```

#### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--capture, -c` | `./my-task` | 录制数据目录 |
| `--model, -m` | `Qwen/Qwen3-VL-2B-Instruct` | HuggingFace 模型名 |
| `--output, -o` | `training_output` | 检查点输出目录 |
| `--epochs, -e` | `40` | 训练轮数 |
| `--lr` | `2e-4` | 学习率 |
| `--batch-size, -b` | `1` | 批大小 |
| `--lora-r` | `32` | LoRA 秩 |
| `--4bit` | — | 启用 4-bit 量化（需 CUDA） |
| `--dry-run, -n` | — | 仅加载数据预览 |

#### 训练配置文件

详细的超参数配置见 [`train_config.yaml`](./train_config.yaml)：

```yaml
model:
  name: "Qwen/Qwen3-VL-2B-Instruct"
  load_in_4bit: false              # Apple Silicon 不支持

lora:
  r: 32                            # LoRA 秩
  lora_alpha: 64                   # 2 × lora_r
  lora_dropout: 0.05
  target_modules: [q_proj, v_proj, k_proj, o_proj]

training:
  num_train_epochs: 40             # 足够让 Loss 降到 1.0 以下
  learning_rate: 2e-4
  warmup_ratio: 0.0                # 极小数据集不需要热身
  gradient_accumulation_steps: 1
  logging_steps: 1
  early_stop_loss: 0.01
```

#### 训练依赖

```bash
pip install trl datasets peft accelerate
```

#### 训练产出

- `training_output/checkpoint-*/` — 每个 epoch 的中间检查点
- `training_output/final/` — 最终合并的 LoRA adapter

---

### 5. 1:1 精确回放（replay_capture.py）

直接回放 `recording.db` 中的原始事件，**完全精确还原**录制操作的坐标和时间间隔：

```bash
# 1:1 原速回放（默认 3 秒倒计时）
python replay_capture.py

# 指定录制目录
python replay_capture.py --capture ./my-task

# 2 倍速回放
python replay_capture.py --speed 2.0

# 慢速回放（0.5 倍速）
python replay_capture.py --speed 0.5

# 预览所有事件（不执行）
python replay_capture.py --dry-run

# 增加准备时间（10 秒倒计时）
python replay_capture.py --delay 10

# 组合使用
python replay_capture.py --capture ./my-task --speed 1.0 --delay 5
```

#### 回放参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--capture, -c` | `./my-task` | 录制数据目录 |
| `--speed, -s` | `1.0` | 回放速度倍率（2.0 = 两倍速） |
| `--delay, -d` | `3.0` | 开始前的倒计时秒数 |
| `--dry-run, -n` | — | 仅打印事件，不执行 |

#### 注意事项

- ⚠️ **需要辅助功能权限**：系统设置 → 隐私与安全 → 辅助功能 → 启用终端/iTerm2
- ⚠️ **屏幕分辨率需一致**：录制的坐标是绝对像素值，分辨率变化会导致点击偏移
- ⚠️ **倒计时期间切换到目标窗口**
- 按 `Ctrl+C` 可随时中断回放

#### 回放依赖

```bash
pip install pynput
```

---

### 6. 模型推理回放（run_model.py）

加载训练好的 LoRA 模型，通过截图推理预测下一步操作并执行：

```bash
# 默认运行（加载 training_output/final）
python run_model.py

# 指定检查点和任务
python run_model.py --checkpoint training_output/final --task my-task

# 增加最大步数
python run_model.py --max-steps 20

# 延长准备时间
python run_model.py --delay 10

# 调整操作间隔
python run_model.py --action-delay 2.0

# 预览模式（不执行操作）
python run_model.py --dry-run

# 完整参数示例
python run_model.py \
    --checkpoint training_output/final \
    --task my-task \
    --max-steps 15 \
    --delay 5 \
    --action-delay 1.0
```

#### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint, -c` | `training_output/final` | LoRA 检查点路径 |
| `--task, -t` | `my-task` | 任务描述/目标 |
| `--max-steps, -s` | `15` | 最大推理步数 |
| `--delay, -d` | `5.0` | 开始前的倒计时秒数 |
| `--action-delay` | `1.0` | 每步操作后的等待时间（秒） |
| `--dry-run, -n` | — | 仅预测不执行 |

#### 推理流程

```
截图 → 模型推理 → 解析动作 → 执行操作 → 等待 → 循环
         ↓
   支持的动作类型:
   - CLICK(x=0.XX, y=0.XX)   点击归一化坐标
   - TYPE(text="...")         输入文本
   - WAIT()                  等待 UI 更新
   - DONE()                  任务完成
```

#### 推理依赖

```bash
pip install pynput Pillow peft transformers torch
```

---

## 📊 两种回放方式对比

| 特性 | 1:1 精确回放 | 模型推理回放 |
|------|-------------|-------------|
| **脚本** | `replay_capture.py` | `run_model.py` |
| **数据源** | `recording.db` 原始事件 | 训练好的 LoRA 模型 |
| **精确度** | ✅ 100% 精确（坐标+时间） | ⚠️ 近似（模型预测） |
| **泛化能力** | ❌ 无（固定坐标回放） | ✅ 有（可适应 UI 变化） |
| **需要 GPU** | ❌ | ⚠️ 推荐（MPS/CUDA） |
| **适用场景** | 固定环境的精确重复 | 不同环境的智能操作 |

---

## 🔧 环境配置

### 使用 pyenv 虚拟环境

```bash
# 激活 OAT 环境
pyenv activate OAT

# 安装全部依赖
pip install pynput Pillow peft transformers torch trl datasets accelerate
```

### macOS 权限设置

回放功能需要 **辅助功能权限**：

1. 打开 **系统设置** → **隐私与安全** → **辅助功能**
2. 点击 `+` 号，添加你使用的终端应用（Terminal / iTerm2 / Warp 等）
3. 确保开关已打开

---

## 🚀 快速上手

```bash
# 1. 录制操作
openadapt capture start --name my-task
# ... 执行操作 ...
openadapt capture stop

# 2. 训练模型
python train_local.py --capture ./my-task --epochs 40

# 3a. 1:1 精确回放
python replay_capture.py --capture ./my-task

# 3b. 或者，基于模型推理回放
python run_model.py --checkpoint training_output/final --task my-task
```
