# MiniLLM 产品需求文档

## 1. 项目概述

MiniLLM 是一个从零开始实现的轻量级语言模型项目，重点放在代码清晰、
结构可理解、工程复杂度可控，而不是追求大规模模型的极限性能。

项目会先实现一个最小可运行的 decoder-only 语言模型主干，然后逐步接入
tokenizer、数据流式读取、训练与推理流程、MoE 架构以及 LoRA 微调能力。

这个项目的核心目标不是和成熟的开源大模型竞争，而是搭建一个足够干净、
足够透明的代码库，让模型从 token 到 logits、loss、checkpoint、generation
的完整路径都可以被直接阅读、理解和修改。

目标硬件约束：

- 训练：目标控制在约 40GB 显存内。
- 推理：目标控制在约 20GB 显存内。
- 初期开发：优先支持单机单卡。

## 2. 项目目标

- 从零实现一个最小可用的 decoder-only Transformer 语言模型。
- 保持代码风格简洁、明确、容易追踪。
- 按阶段推进，每个版本都应该有独立可运行的成果。
- 在加入复杂功能前，先跑通基础训练和文本生成。
- 在基础主干稳定后，再引入 MoE 和 LoRA。
- 在小规模实验范围内控制显存占用，避免过早陷入复杂分布式训练。

## 3. 非目标

- 不追求达到生产级开源大模型的性能。
- 初期不构建复杂的分布式训练框架。
- 初期不支持所有 tokenizer、数据格式、量化方式或服务部署后端。
- 初期不实现 RLHF、DPO、偏好优化等复杂对齐流程。
- 不为了抽象而抽象，核心模型逻辑应尽量直接、可读。

## 4. 目标用户

这个项目主要面向希望通过亲手实现来理解 LLM 内部机制的开发者或研究学习者。
项目应方便用户查看和修改模型结构、训练循环、生成逻辑，以及后续的 MoE
路由行为。

## 5. 版本路线

### v0：MVP 主干

阶段目的：实现最简单、能跑通的模型主干。

范围：

- decoder-only Transformer 架构。
- token embedding。
- 最小形式的位置处理。
- multi-head self-attention。
- feed-forward network，也就是 FFN。
- residual connection 和 normalization。
- 用于 next-token prediction 的 LM head。
- 最小 forward pass 和 loss 计算。
- 最小 generation 函数。

预期结果：

- 模型可以接收 token ID 并输出 logits。
- 模型可以计算 next-token cross-entropy loss。
- 模型可以在合成数据或玩具数据上完成 tiny overfit 测试。
- 模型可以基于一段 token ID prompt 生成后续 token ID。

### v1：基础训练与推理流程

阶段目的：让项目具备真实小规模训练和基础推理的能力。

范围：

- 接入开源 tokenizer。
- 支持 dataset loading 和 data streaming。
- 实现 causal attention mask。
- 接入 RoPE 位置编码。
- 在可用时接入 FlashAttention。
- 支持 checkpoint 保存与加载。
- 实现带日志的训练循环。
- 实现基础 validation loop。
- 使用配置文件管理模型和训练参数。
- 提供训练与生成的 CLI 入口。

预期结果：

- 小模型可以在真实文本数据上训练。
- checkpoint 可以保存、恢复训练。
- 训练后的 checkpoint 可以被加载并用于文本生成。
- 训练和推理主路径足够稳定，可以支撑后续架构实验。

### v2：MoE 架构

阶段目的：在 dense baseline 清晰稳定的基础上，引入 Mixture-of-Experts。

范围：

- 将标准 FFN block 替换为可选的 MoE FFN block。
- 实现用于 expert selection 的 router。
- 初始版本使用 top-1 routing。
- top-1 稳定后，再考虑可选的 top-2 routing。
- 实现 expert load-balancing auxiliary loss。
- 输出 routing statistics，便于调试。
- 通过配置切换 dense FFN 和 MoE FFN。

预期结果：

- dense 模型和 MoE 模型复用同一套训练、推理流程。
- 小规模 MoE 实验中不出现明显 expert collapse。
- 训练时可以观察 expert 的路由分布。
- 可以在相近 active parameter budget 下比较 dense 与 MoE 版本。

### v3：LoRA 微调

阶段目的：在已有基础模型之上支持参数高效微调。

范围：

- 为 attention projection 接入 LoRA module。
- 可选地为 FFN 或 MoE expert projection 接入 LoRA。
- LoRA 训练时冻结 base model 参数。
- LoRA adapter 权重与 base checkpoint 分开保存和加载。
- 如有需要，支持将 LoRA 权重 merge 回模型用于推理。

预期结果：

- base model 可以用较少的可训练参数完成微调。
- LoRA adapter 可以保存、加载、切换。
- 项目可以支持轻量级指令微调或领域适配实验。

## 6. 架构方向

基础模型保持标准 decoder-only Transformer 结构：

- token embedding。
- 多层 Transformer block。
- causal self-attention。
- feed-forward network。
- normalization。
- final LM head。

v1 之后推荐的 block 结构：

```text
input token ids
  -> token embedding
  -> TransformerBlock x N
       -> RMSNorm
       -> causal self-attention with RoPE
       -> residual
       -> RMSNorm
       -> FFN or MoE-FFN
       -> residual
  -> final norm
  -> LM head
  -> logits
```

MoE 应作为 FFN 路径的可替换实现接入，而不是另起一套重复的模型代码。

## 7. 显存与规模预期

项目应围绕小规模和中等规模实验设计，而不是一开始就面向大规模生产训练。

初期较现实的模型规模：

- v0/v1 dense 实验：约 10M 到 500M 参数。
- 更大的 dense 实验：在显存允许时可尝试约 0.5B 到 1.5B 参数。
- v2 MoE 实验：active parameters 约 0.5B 到 1.5B，总参数可尝试约 1B 到 3B。

训练显存控制方式：

- 使用 bf16 或 fp16 mixed precision。
- 使用 gradient accumulation。
- 后续按需加入 activation checkpointing。
- 可用时使用 FlashAttention。
- 默认使用保守的 batch size 和 sequence length。

推理显存后续可以通过以下方式优化：

- 减小 batch size。
- 管理 KV cache。
- 后续可选支持 int8 或 int4 quantization。

## 8. 质量标准

每个阶段都应该有一个小而明确的可运行验收条件：

- v0：在一个 tiny synthetic batch 上过拟合，并能生成 token ID。
- v1：在小文本数据集上训练，保存 checkpoint，重新加载后通过 tokenizer decode
  生成文本。
- v2：训练一个 tiny MoE 模型，并能观察到非平凡的 expert routing。
- v3：使用 LoRA 微调，同时保持绝大多数 base model 参数冻结。

早期测试应优先覆盖 smoke test 和关键路径，而不是追求大而全的测试体系。

## 9. 建议目录结构

```text
minillm/
  PRD.md
  README.md
  pyproject.toml
  configs/
  minillm/
    __init__.py
    model.py
    attention.py
    ffn.py
    moe.py
    lora.py
    tokenizer.py
    data.py
    train.py
    generate.py
    checkpoint.py
  scripts/
    train.py
    generate.py
  tests/
```

目录结构可以随着项目推进调整。基本原则是：核心模型代码应该容易找到，
也应该容易阅读。

## 10. 主要风险

- 从零训练在缺少大规模高质量数据时，对话能力会比较有限。
- MoE 在 batch size 较小或路由不稳定时，可能出现 expert collapse。
- 训练过程可能很快受限于显存，而不是模型结构本身。
- 过早加入太多功能会让代码库变得难以理解。
- tokenizer 和数据选择会显著影响最终模型的表现。

## 11. 成功定义

如果这个项目最终能提供一个清晰、可运行、可扩展的 mini LLM 实现，就算达成目标。
具体来说，它应该能够：

- 端到端训练一个小型 decoder-only 模型。
- 从 checkpoint 加载模型并生成文本。
- 在不重写整套代码的情况下，从 dense FFN 演进到 MoE。
- 通过 LoRA 支持轻量级微调。
- 保持足够透明，让读者可以追踪从 tokens 到 logits、loss、checkpoint、
  generation 的完整路径。

