# 阶段路线

## v0：MVP 主干

- 实现基础 Transformer block。
- 实现 FFN、attention、normalization。
- 支持 toy loss 和 toy generation。
- 提供 `scripts/stage_tests/tiny_overfit.py`，验证固定小 batch 可以被过拟合。

## v1：训练与推理主干

- 接入 tokenizer。
- 接入 data streaming。
- 加入 causal mask、RoPE、FlashAttention。
- 支持 checkpoint 保存与恢复。
- 提供训练与生成脚本。

## v2：MoE

- 将 FFN 替换为可选 MoE-FFN。
- 实现 top-1 router。
- 记录 expert routing statistics。
- 加入 load balancing loss。

## v3：LoRA

- 为 attention projection 接入 LoRA。
- 支持冻结 base model。
- 支持 adapter 保存、加载与切换。
