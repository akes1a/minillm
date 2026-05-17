# MiniLLM

MiniLLM 是一个从零开始实现的轻量级 LLM 项目。项目会按照四个阶段推进：

- v0：实现最小 decoder-only Transformer + FFN 主干。
- v1：接入 tokenizer、data streaming、causal mask、RoPE、FlashAttention、checkpoint。
- v2：在 FFN 路径上接入 MoE。
- v3：加入 LoRA 微调能力。

当前目录主要是结构化骨架，后续实现应优先保持代码清晰、依赖克制、模块边界明确。

## 目录概览

```text
minillm/
  configs/          # 各阶段配置
  docs/             # 设计记录和阶段说明
  minillm/          # Python 源码包
  scripts/          # 命令行入口脚本
  tests/            # smoke test 和单元测试
  PRD.md            # 项目需求文档
  README.md         # 项目说明
  pyproject.toml    # Python 项目配置
```

## 开发原则

- 先跑通主路径，再加入高级能力。
- 每个阶段都保留可独立验证的 smoke test。
- MoE 和 LoRA 作为可选能力接入，不破坏 dense baseline。
- 核心模型逻辑优先可读，不为了过早优化牺牲清晰度。

