# Spatial-MLLM backbone transport

工作目录：`/home/u2024311136/SandMem`。用户于2026-11-03批准官方默认
`Diankun/Spatial-MLLM-v1.1-Instruct-135K`，revision
`635b63733ed24f76b2f1a533dd5bc32836ea6c53`。仅替换backbone，不改D116方法。

## 代码与使用

`sandbox_vlm/backbones/spatial_transport.py`提供一个共享冻结权重的
`SpatialBackbone`，暴露`discovery`、`answerer`、`relevance`和`coverage`。
发现/批发现/重复停止、QA生成及解析、Yes/No差值、4×4/64维覆盖描述和
严格像素核验的视觉特征复用沿用原实现。未修改任何原冻结方法文件、
旧默认配置、身份规则、SAM/VGGT几何实现、提示、输出上限或评分。

```python
import json
from pathlib import Path
from sandbox_vlm.backbones.spatial_transport import SpatialBackbone

config = json.loads(Path('configs/spatial135k.json').read_text())
backbone = SpatialBackbone(config, '/path/to/complete/pinned/Spatial-MLLM')
discovery, answerer = backbone.discovery, backbone.answerer
scorer, encoder = backbone.relevance, backbone.coverage
```

`configs/spatial135k.json`与`configs/reference_qwen.json`只差两处MLLM的
model_id/revision，共四个值。历史Qwen入口保留；不能把它们直接读入的
Qwen冻结实例重标为完整Spatial感知。新调用者显式注入上述对象，不能
继续创建原Qwen对象。`mainline.py`依然只是历史兼容接口，非D116正式入口。

## 必要接口差异

- 使用权重自带Qwen2.5 processor/chat template，经官方视觉工具处理，
  同时传入视觉patch和额外空间流。静态图各自重复为两个时间槽，沙盒
  保持独立image块；不将沙盒与源帧拼为连续视频。
- 独立提示的batch边界、左padding和每条输出解析不变；嵌套空间tensor
  随token tensor一起搬到设备。
- Spatial无`logits_to_keep`，保留完整forward后读取同样的最后位置
  Yes/No logits，额外logits成本如实记录。
- Spatial没有Qwen3 DeepStack。覆盖特征使用同一Spatial的2D视觉塔，
  保持原池化算法和精确像素/网格复用；后续相关性forward仍执行实际
  空间编码器与连接器。CPU图像转换重新执行，不借用Qwen3转换缓存。
- 加载时隔离官方内部`vggt.*`导入，随后恢复外部VGGT命名空间和路径。
  不以内部空间编码器替代本方法的外部VGGT几何估计。
- 用户明确批准处理后宽度1280→1288的原生28像素网格对齐。仅在原始
  图像边长仍≤1280时允许处理后≤1288；其他额度原样保留。原图像素不改，
  超过1280的源图仍拒绝。该例外在每次QA的`processor_alignment`中披露。

## 接线验收入口

`scripts/check_spatial_backbone.py --source-audit <既有CPU输入审计目录>
--output <全新目录>`默认只做CPU准备；加`--gpu --official-root <完整官方仓库>`
才加载模型。只读取输入/消息/像素/查询水位，不读答案、评分或旧候选。
保存实际输入tensor、提示、原始输出、解析、实际空间编码器hook和显存。

GPU探针包含覆盖、相关性、两条独立发现；稀疏既有两阶段输入再现场
生成D065原帧候选，调用原核验与决定规则，密集输入直接联合回答。
原始生成上限和笔记边界保持，不补答、不调参、不自动全量。
本探针复用旧Qwen的结构和几何，因此严格标记固定旧记忆的QA替换接线，
不是完整Spatial记忆重建、正式baseline或新模型得分。

## 环境与状态

独立环境位于`work/envs/spatial135k`；官方完整源码只读复制于
`work/spatial-mllm-official`，commit
`9fc47382c7bc5ab52951e6e2e64db08fca0948ee`。仓库vendor中的Spatial文件
仍仅是CPU fixture。运行依赖见`requirements-spatial.txt`，不改旧环境。

全部验收、下载失败、CPU入口失败保存在`work/migration_135k_20261103/`。
CPU准备成功不等于GPU验收或完整迁移完成。最终状态以该目录status.json
及各独立输出目录的completion/failure.json为准。

不修改AGENTS.md、PLAN.md或Proposal；原Qwen结果与旧冻结文件全部保留。

## 实际验收结论

已完成backbone适配和真实A100前向，**输出契约验收失败，不能宣称完整
Spatial迁移成功或作为正式默认运行**。未为适应新模型修改提示或解析。

| 固定输入 | 模型请求 | 发现输出 | QA输出格式 | 联合实际空间图像 |
| --- | ---: | --- | --- | ---: |
| 稀疏，17147 | 4：筛帧/批发现/候选/核验 | 2/2失败 | 原帧候选`1`无有效末行；核验为`FINAL_ANSWER: 1`，不符合原题A–D要求 | 9，含沙盒 |
| 密集，17148 | 3：筛帧/批发现/直接联合QA | 2/2失败 | `2`可解析为有限数值；未读真值、未评分 | 33，含沙盒 |

两次作业均COMPLETED/0:0，分别2:33/1:42；首次17131仅PyTorch库路径
启动失败，7秒、0模型调用，独立保留。共7次模型请求，另有2次覆盖
视觉塔编码；0新增外部SAM/VGGT/track/训练/模型补答。Spatial内部空间
编码属于MLLM前向成本，不与外部VGGT混淆。4次逻辑发现均失败，其中
两个输出`D`、两个输出`2`，不能用于重新构建实例记忆。
更正：此前本段将稀疏发现短答误写为`1`；`1`来自稀疏QA，发现的原始
输出及机器可读status.json始终是`D`，未修改任何运行产物。

稀疏/密集GPU峰值allocated约15.64/16.43GiB，reserved约16.97/18.12GiB；
实际设备A100-PCIE-40GB。这只覆盖两个固定输入和小发现/筛帧batch，
不能认证全量显存上界或A30可用。所有三个权重分片完整SHA256通过。

原147个冻结文件、98个依赖文件哈希保持；110项CPU回归通过（原101项
加9项适配测试，包含BF16复用检查和VGGT命名空间恢复）。CPU稀疏/密集
独立batch与单条处理一致；原始1280边长拒绝记录和用户批准1288之后的
独立成功记录都保留。

环境核心包为torch2.6.0/cu124、torchvision0.21.0、transformers4.51.3、
flash-attn2.7.4.post1、deepspeed0.16.9和triton3.2.0。兼容依赖通过独立
环境的`.pth`只读引用原已安装依赖；原环境未安装/卸载任何包。启动器
显式加入CUDA库搜索路径。这个验收环境不是从零安装认证，完整依赖
快照在`environment_freeze.txt`。正式比较仍需另行固定完整运行环境。

机器可读结论：`work/migration_135k_20261103/status.json`。GPU实际输入、
原始失败、源代码封存、输出、hook、资源与启动失败都在同目录；
`pre_gpu_manifest_04.json`对应实际完成的两次GPU接线源码。

本次未启动全量、baseline或提示/阈值搜索，未生成新准确率结果；
`project.json`的完整迁移状态仍保持false。
