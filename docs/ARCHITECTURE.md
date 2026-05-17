# 架构说明

MiniLLM 的核心结构是 decoder-only Transformer。项目会先实现 dense FFN 版本，
再将 FFN 路径扩展为可选的 MoE，最后接入 LoRA 微调。

推荐的主路径：

```text
token ids
  -> embedding
  -> TransformerBlock x N
       -> attention
       -> FFN / MoE-FFN
  -> final norm
  -> LM head
  -> logits
```

模块边界：

- `minillm/core/`：基础模型结构。
- `minillm/data/`：数据读取与 batch 构造。
- `minillm/tokenizers/`：tokenizer 适配层。
- `minillm/training/`：训练循环、优化器、checkpoint。
- `minillm/inference/`：生成与推理逻辑。
- `minillm/moe/`：MoE router、expert、辅助 loss。
- `minillm/lora/`：LoRA module 与 adapter 管理。

