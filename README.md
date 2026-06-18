# Visual-Speech-Recognition-V2

中文唇语识别项目第二版：基于 CAS-VSR-W1k / LRW-1000 大规模中文唇语数据集，构建端到端视觉语音识别模型。模型采用 **3D CNN 时空前端 + SE-ResNet18 空间特征提取器 + Bi-GRU 时序建模模块**，并融合 **Word Boundary 词边界信息**、**Mixup 数据增强** 与 **Label Smoothing 标签平滑**，用于提升自然场景下中文词级唇语识别的鲁棒性与泛化能力。

> Visual Speech Recognition V2 for Chinese lip reading.
> This project builds an end-to-end visual speech recognition framework based on 3D CNN, SE-ResNet18, Bi-GRU, Word Boundary, Mixup and Label Smoothing.

---

## 目录

* [项目简介](#项目简介)
* [项目特点](#项目特点)
* [模型结构](#模型结构)
* [数据集说明](#数据集说明)
* [仓库结构](#仓库结构)
* [环境配置](#环境配置)
* [数据预处理](#数据预处理)
* [训练方法](#训练方法)
* [验证与测试](#验证与测试)
* [实验结果](#实验结果)
* [运行注意事项](#运行注意事项)
* [后续计划](#后续计划)
* [项目成员](#项目成员)
* [License](#license)

---

## 项目简介

视觉语音识别（Visual Speech Recognition, VSR），也称为唇语识别，旨在仅根据说话人的面部与唇部运动序列推断其表达的文本内容。与传统音频语音识别不同，唇语识别不依赖声音信号，因此在强背景噪声、音频缺失、隐私保护、人机交互和边缘智能设备等场景中具有重要应用价值。

本项目围绕中文词级唇语识别任务，基于 CAS-VSR-W1k（原 LRW-1000）数据集完成模型训练与实验分析。项目重点包括：

1. 将唇部视频序列统一预处理为单通道灰度帧序列；
2. 使用 3D CNN 提取局部短时空运动特征；
3. 使用融合 SE Block 的 ResNet-18 提取深层空间语义特征；
4. 使用 Bi-GRU 建模长距离唇部运动时序依赖；
5. 引入 Word Boundary 词边界信息，辅助模型定位有效发音区间；
6. 使用 Mixup 与 Label Smoothing 缓解过拟合并提升泛化性能；
7. 通过消融实验与超参数调参实验验证各模块有效性。

---

## 项目特点

* **端到端视觉语音识别框架**
  输入为唇部视频帧序列，输出为 1000 类中文词汇 / 短语的分类结果。

* **3D CNN 时空前端**
  在时间和空间维度上同时卷积，捕获嘴唇开合、收缩、圆唇、展唇等短时运动特征。

* **SE-ResNet18 空间特征提取器**
  在 ResNet 基本残差块中加入 SE 通道注意力机制，使模型能够自适应增强与唇部形状和局部运动纹理相关的特征通道。

* **Bi-GRU 时序建模**
  使用三层双向 GRU 同时建模历史上下文和未来上下文，捕获完整发音过程中的长距离时序依赖。

* **Word Boundary 词边界融合**
  将每一帧是否处于目标词发音区间内的边界标记与视觉特征拼接，减少发音前后冗余帧对模型判断的干扰。

* **Mixup + Label Smoothing 正则化**
  训练阶段引入样本混合增强和软标签约束，降低模型对单一样本和 one-hot 标签的过度拟合。

* **适合后续边缘端部署优化**
  训练阶段的 Mixup、Label Smoothing 和 Weight Decay 不改变推理阶段网络结构，因此不会显著增加部署阶段计算开销。

---

## 模型结构

整体网络按照如下流程进行特征提取与分类：

```text
Input Lip Video
        │
        ▼
3D CNN Frontend
        │
        ▼
SE-ResNet18 Spatial Encoder
        │
        ▼
Frame-level Visual Feature: B × T × 512
        │
        ├── Word Boundary: B × T × 1
        │
        ▼
Feature Concatenation: B × T × 513
        │
        ▼
3-layer Bi-GRU
        │
        ▼
Temporal Feature: B × T × 2048
        │
        ▼
Dropout + Linear Classifier
        │
        ▼
Temporal Average Pooling
        │
        ▼
Class Logits: B × 1000
```

### 1. 输入格式

模型输入为裁剪后的单通道灰度唇部视频序列：

```text
X ∈ R^(B × T × 1 × 88 × 88)
```

其中：

* `B`：batch size；
* `T`：视频帧数；
* `1`：灰度通道；
* `88 × 88`：裁剪后的唇部图像尺寸。

### 2. 3D CNN 前端

`video_cnn.py` 中的 3D CNN 前端用于捕获局部时空运动特征：

```text
Conv3D: kernel_size = (5, 7, 7)
stride = (1, 2, 2)
padding = (2, 3, 3)
```

该层在时间轴上不降采样，以保留完整帧序列；在空间维度上进行下采样，降低后续 ResNet 的计算量。

### 3. SE-ResNet18 空间编码器

每一帧经过 3D CNN 前端后，被送入 ResNet-18 进行空间特征提取。模型在基本残差块中加入 SE Block，通过通道注意力机制对不同特征通道进行自适应重标定。

SE Block 的基本思想为：

```text
Global Average Pooling → Channel Bottleneck → Sigmoid Gate → Channel Re-weighting
```

最终每一帧被编码为 512 维视觉特征：

```text
X_spatial ∈ R^(B × T × 512)
```

### 4. Word Boundary 融合

当启用 `--border true` 时，模型会将 Word Boundary 与视觉特征在最后一维进行拼接：

```text
X_tilde = concat(X_spatial, B_duration)
```

拼接后每个时间步的输入维度从 512 扩展为 513：

```text
X_tilde ∈ R^(B × T × 513)
```

Word Boundary 用于显式提示模型：当前帧是否处于目标词汇的有效发音区间内。

### 5. Bi-GRU 时序建模

时序建模模块使用三层双向 GRU：

```text
input_dim = 513  # 启用 Word Boundary 时
hidden_dim = 1024
num_layers = 3
bidirectional = True
dropout = 0.2
```

双向 GRU 输出维度为：

```text
B × T × 2048
```

### 6. 分类头

分类头对每一个时间步的 Bi-GRU 输出进行线性映射，然后沿时间维度求平均，得到整段视频的分类 logits：

```text
logits = mean(Linear(BiGRU_output), dim=1)
```

最终输出为：

```text
B × 1000
```

对应 CAS-VSR-W1k / LRW-1000 的 1000 个中文词汇 / 短语类别。

---

## 数据集说明

本项目使用 CAS-VSR-W1k 数据集，即 LRW-1000 的更新版本。该数据集来自真实自然场景，具有较强的复杂性和挑战性。

数据集特点：

* 包含 1000 个中文词汇 / 短语类别；
* 包含 2000 多名说话人；
* 总样本规模超过 700,000 条；
* 视频帧率统一为 25 fps；
* 覆盖不同姿态、年龄、性别、语速、光照和分辨率；
* 官方提供训练集、验证集和测试集划分；
* 官方还提供 pose、resolution、length 等不同难度子集。

> 注意：由于数据集体积较大且可能涉及数据使用协议，本仓库不直接包含原始数据集。请自行根据 CAS-VSR-W1k / LRW-1000 的官方说明获取数据。

---

## 仓库结构

当前仓库主要文件说明如下：

```text
Visual-Speech-Recognition-V2/
│
├── README.md
│   └── 项目说明文档
│
├── train.py
│   └── 模型训练与验证入口
│
├── model.py
│   └── 整体 VideoModel 定义，包含 VideoCNN、Bi-GRU、分类头等
│
├── video_cnn.py
│   └── 3D CNN 前端、SE-ResNet18 空间特征提取网络
│
├── dataset.py
│   └── LRW1000 数据加载类，包含 TurboJPEG 灰度解码、随机裁剪、中心裁剪等
│
├── prepare_lrw1000.py
│   └── 根据官方标注文件生成训练所需的 pkl 数据
│
├── mp_preprocess_lrw1000.py
│   └── 多进程数据预解码脚本，用于提升后续训练读取效率
│
├── log-*.out
│   └── Mixup、SE、Word Boundary、学习率等实验日志
│
└── slurm-*.out
    └── 集群训练任务日志
```

---

## 环境配置

### 1. 创建 Python 环境

推荐使用 Conda：

```bash
conda create -n lipread python=3.8 -y
conda activate lipread
```

也可以使用 Python venv：

```bash
python -m venv lipread
source lipread/bin/activate
```

### 2. 安装 PyTorch

请根据自己的 CUDA 版本安装对应的 PyTorch。示例：

```bash
pip install torch torchvision torchaudio
```

如果使用 CUDA，请参考 PyTorch 官网选择合适命令。

### 3. 安装常用依赖

```bash
pip install numpy opencv-python tqdm PyTurboJPEG
```

Ubuntu 系统中，`PyTurboJPEG` 可能需要系统库支持：

```bash
sudo apt-get update
sudo apt-get install -y libturbojpeg
```

### 4. 可能需要的依赖模块

当前训练脚本中可能涉及以下自定义模块或工具：

```text
LSR.py
utils.py
cvtransforms.py
```

如果本地运行时报错：

```text
ModuleNotFoundError: No module named 'LSR'
ModuleNotFoundError: No module named 'utils'
ModuleNotFoundError: No module named 'cvtransforms'
```

请根据实际代码结构补充对应文件，或修改导入路径。例如，如果使用当前仓库中的 `dataset.py`，可以将 `train.py` 中的数据集导入方式调整为本地实际路径。

---

## 数据预处理

### 1. 原始数据目录

请先下载并解压 CAS-VSR-W1k / LRW-1000 数据集。建议整理为如下结构：

```text
data/
├── lip_images/
│   ├── class_or_video_folder_1/
│   ├── class_or_video_folder_2/
│   └── ...
│
└── labels/
    ├── trn_1000.txt
    ├── val_1000.txt
    └── tst_1000.txt
```

标注文件中每行样本通常包含：

```text
Image_Folder_Name, Class_Name_Chinese, Class_Name_PinYin, Start_Time, End_Time
```

### 2. 修改路径

运行前需要根据本机路径修改以下脚本中的硬编码路径。

在 `prepare_lrw1000.py` 中修改：

```python
self.data_root = '/path/to/lip_images'
```

以及：

```python
label_files = {
    'trn': '/path/to/labels/trn_1000.txt',
    'val': '/path/to/labels/val_1000.txt',
    'tst': '/path/to/labels/tst_1000.txt'
}
```

在 `dataset.py` 中修改：

```python
base_dir = '/path/to/LRW1000_Public_pkl_jpeg'
```

### 3. 生成 pkl 数据

运行：

```bash
python prepare_lrw1000.py
```

该脚本会根据官方标注文件读取唇部图像序列，并生成训练所需的 `.pkl` 文件。每个 pkl 文件中主要包含：

```python
{
    "video": encoded_video_frames,
    "label": class_id,
    "duration": word_boundary
}
```

其中：

* `video`：视频帧序列；
* `label`：类别编号；
* `duration`：Word Boundary 词边界标记。

### 4. 可选：多进程预解码

如果希望进一步减少训练时的数据解码开销，可以使用：

```bash
python mp_preprocess_lrw1000.py
```

运行前同样需要修改脚本中的路径：

```python
SRC_BASE = "/path/to/source_pkl"
DST_BASE = "/path/to/decoded_dataset"
```

---

## 训练方法

训练入口为：

```bash
python train.py
```

主要参数如下：

| 参数               | 含义                        |
| ---------------- | ------------------------- |
| `--gpus`         | 使用的 GPU 编号，例如 `0` 或 `0,1` |
| `--lr`           | 初始学习率                     |
| `--batch_size`   | 批大小                       |
| `--n_class`      | 类别数，LRW-1000 为 `1000`     |
| `--num_workers`  | DataLoader 进程数            |
| `--max_epoch`    | 最大训练轮数                    |
| `--test`         | 是否只测试                     |
| `--weights`      | 测试或继续训练时加载的模型权重           |
| `--save_prefix`  | 模型保存路径前缀                  |
| `--dataset`      | 数据集类型，当前可设为 `lrw1000`     |
| `--border`       | 是否启用 Word Boundary        |
| `--mixup`        | 是否启用 Mixup                |
| `--label_smooth` | 是否启用 Label Smoothing      |
| `--se`           | 是否启用 SE Block             |

### 推荐训练命令

```bash
python train.py \
  --gpus 0 \
  --lr 5e-4 \
  --batch_size 64 \
  --n_class 1000 \
  --num_workers 8 \
  --max_epoch 120 \
  --test false \
  --save_prefix checkpoints/final/model \
  --dataset lrw1000 \
  --border true \
  --mixup true \
  --label_smooth true \
  --se true
```

### 推荐配置

| 配置项                     | 推荐值                           |
| ----------------------- | ----------------------------- |
| Model                   | 3D CNN + SE-ResNet18 + Bi-GRU |
| Word Boundary           | `true`                        |
| Mixup                   | `true`                        |
| Mixup alpha             | `0.2`                         |
| Label Smoothing         | `true`                        |
| Label Smoothing epsilon | `0.1`                         |
| Optimizer               | Adam                          |
| Weight Decay            | `1e-4`                        |
| LR Scheduler            | CosineAnnealingLR             |
| Learning Rate           | `3e-4` 到 `5e-4`               |
| Max Epoch               | `120`                         |

> 当前代码中 `Mixup alpha` 和 `Weight Decay` 位于 `train.py` 内部。如果需要调参，请直接修改源码中的对应变量。

---

## 验证与测试

如果已有训练好的权重文件，可以使用 `--test true` 进入测试 / 验证模式：

```bash
python train.py \
  --gpus 0 \
  --lr 5e-4 \
  --batch_size 64 \
  --n_class 1000 \
  --num_workers 8 \
  --max_epoch 120 \
  --test true \
  --weights checkpoints/final/model_iter_xxx_epoch_xxx_v_acc_xxx.pt \
  --save_prefix checkpoints/final/model \
  --dataset lrw1000 \
  --border true \
  --mixup false \
  --label_smooth false \
  --se true
```

测试阶段通常不需要启用 Mixup 和 Label Smoothing，因为它们只用于训练阶段的损失构造。

---

## 实验结果

### 1. 消融实验结果

| 实验设置     | 启用模块                              | Best Acc. | Last Acc. |
| -------- | --------------------------------- | --------: | --------: |
| Baseline | 无额外模块                             |   45.694% |   44.385% |
| SE       | SE Block                          |   45.850% |   44.683% |
| SE + WB  | SE Block + Word Boundary          |   53.704% |   53.267% |
| Final    | SE + WB + Mixup + Label Smoothing |   56.125% |   55.675% |

从实验结果可以看出：

* 单独加入 SE Block 后，模型性能略有提升；
* 加入 Word Boundary 后，验证准确率显著提升，说明词边界信息对自然场景唇语识别非常关键；
* 在 SE 与 Word Boundary 的基础上继续加入 Mixup 和 Label Smoothing 后，模型泛化能力进一步提升；
* 最终模型在验证集上取得最高性能，并在训练后期保持较稳定表现。

### 2. Mixup alpha 调参结果

| Mixup alpha | Best Acc. |
| ----------: | --------: |
|         0.1 |  56.2476% |
|         0.2 |  56.0964% |
|         0.3 |  56.4472% |

从单次最优验证准确率看，`alpha = 0.3` 略高；但综合样本扰动强度、语义可辨识性和训练稳定性，本项目默认采用：

```text
alpha = 0.2
```

### 3. 学习率调参结果

| Learning Rate | Best Acc. |
| ------------: | --------: |
|          1e-4 |  55.9258% |
|          3e-4 |  56.0964% |
|          5e-4 |  56.1759% |

实验表明，过小的学习率会限制模型在固定训练轮数内的优化效率；在当前训练设置下，`3e-4` 到 `5e-4` 是较合理区间，其中 `5e-4` 在该组实验中表现最好。

---

## 运行注意事项

### 1. 数据路径需要手动修改

当前代码中存在若干硬编码路径，例如：

```python
/share/home/dwj/...
/lip/decoded_dataset
```

请在运行前修改为自己的本地路径。

### 2. 当前仓库未包含数据集

由于 CAS-VSR-W1k / LRW-1000 数据集体积较大，本仓库不直接提供原始数据。请自行准备数据并按照本项目格式进行预处理。

### 3. 当前仓库未包含完整训练权重

如果需要复现实验结果，需要重新训练模型，或自行添加训练好的 `.pt` 权重文件。

### 4. 导入路径可能需要调整

如果直接运行出现导入错误，请检查：

```python
from utils import LRW1000_Dataset as Dataset
from LSR import LSR
from .cvtransforms import *
```

根据本地文件结构修改为正确导入路径。

### 5. Mixup 与 Label Smoothing 只作用于训练阶段

这两个策略不会改变推理阶段网络结构，因此测试和部署时不需要额外计算开销。

---

## 后续计划

* [ ] 补充完整依赖文件与 requirements.txt；
* [ ] 移除脚本中的硬编码绝对路径，改为命令行参数或配置文件；
* [ ] 提供单视频推理脚本；
* [ ] 提供训练好的模型权重下载链接；
* [ ] 增加实时摄像头唇语识别 Demo；
* [ ] 增加边缘计算设备部署说明；
* [ ] 尝试模型剪枝、量化和轻量化部署；
* [ ] 支持 RK3588、Jetson、树莓派等嵌入式平台；
* [ ] 增加可视化结果，包括预测类别、置信度、唇部区域和词边界信息。

---

## 项目成员

* 宋志宸
* 邓文杰
* 吴舒一

---

## 致谢

本项目使用 CAS-VSR-W1k / LRW-1000 中文唇语识别数据集，并参考了视觉语音识别领域中常见的 3D CNN、ResNet、GRU、SE Block、Mixup 和 Label Smoothing 等方法。感谢相关数据集与开源社区对中文唇语识别研究的支持。

---

## License

当前仓库暂未声明开源许可证。若需要公开发布、二次开发或商业使用，请先补充 `LICENSE` 文件并明确授权范围。
