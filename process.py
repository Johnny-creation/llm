from transformers import AutoTokenizer
import torch
from model.model_minimind import MiniMindForCausalLM

# 加载tokenizer
tokenizer_path = "/home/zhaojiongning/llm/minimind/model"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# 加载模型
model_path = "/home/zhaojiongning/llm/minimind/minimind-3"
model = MiniMindForCausalLM.from_pretrained(model_path)

# 输入文本
text = "我爱你"

print("=== 完整流程演示 ===")
print(f"输入文本: {text}")
print()

# 步骤1: 文本 -> tokens
tokens = tokenizer.tokenize(text)
print("步骤1: Tokenization")
print(f"Tokens: {tokens}")
print(f"Tokens数量: {len(tokens)}")
print()

# 步骤2: tokens -> token ids
token_ids = tokenizer.encode(text, add_special_tokens=False)
print("步骤2: Encoding to Token IDs")
print(f"Token IDs: {token_ids}")
print(f"Token IDs数量: {len(token_ids)}")
print()

# 步骤3: 转换为tensor，模拟batch
input_ids = torch.tensor([token_ids])
print("步骤3: 转换为Tensor")
print(f"Input IDs tensor shape: {input_ids.shape} (B: batch_size={input_ids.shape[0]}, L: seq_len={input_ids.shape[1]})")
print(f"Input IDs: {input_ids}")
print()

# 步骤4: attention_mask (假设没有padding)
attention_mask = torch.ones_like(input_ids)
print("步骤4: Attention Mask")
print(f"Attention Mask shape: {attention_mask.shape} (B: batch_size={attention_mask.shape[0]}, L: seq_len={attention_mask.shape[1]})")
print(f"Attention Mask: {attention_mask}")
print()

# 步骤5: Embedding
with torch.no_grad():
    embeddings = model.model.embed_tokens(input_ids)
    print("步骤5: Embedding")
    print(f"Embeddings shape: {embeddings.shape} (B: batch_size={embeddings.shape[0]}, L: seq_len={embeddings.shape[1]}, D: hidden_size={embeddings.shape[2]})")
    print(f"Embeddings (first token, first 10 dims): {embeddings[0, 0, :10]}")
    print()

    # 步骤6: 通过模型层
    hidden_states = embeddings
    position_embeddings = (model.model.freqs_cos[:input_ids.shape[1]], model.model.freqs_sin[:input_ids.shape[1]])
    print("步骤6: Position Embeddings")
    print(f"Cos shape: {position_embeddings[0].shape} (L: seq_len={position_embeddings[0].shape[0]}, D: head_dim={position_embeddings[0].shape[1]})")
    print(f"Sin shape: {position_embeddings[1].shape} (L: seq_len={position_embeddings[1].shape[0]}, D: head_dim={position_embeddings[1].shape[1]})")
    print()

    # 经过所有transformer层
    for i, layer in enumerate(model.model.layers):
        hidden_states, _ = layer(hidden_states, position_embeddings)
        print(f"步骤7.{i+1}: Transformer Layer {i+1}")
        print(f"Hidden states shape after layer {i+1}: {hidden_states.shape} (B, L, D)")
        print(f"Hidden states (first token, first 5 dims after layer {i+1}): {hidden_states[0, 0, :5]}")
        print()

    # 步骤8: 最终归一化
    hidden_states = model.model.norm(hidden_states)
    print("步骤8: Final Normalization")
    print(f"Final hidden states shape: {hidden_states.shape} (B, L, D)")
    print(f"Final hidden states (first token, first 5 dims): {hidden_states[0, 0, :5]}")
    print()

    # 步骤9: Language Model Head
    logits = model.lm_head(hidden_states)
    print("步骤9: Language Model Head")
    print(f"Logits shape: {logits.shape} (B: batch_size={logits.shape[0]}, L: seq_len={logits.shape[1]}, V: vocab_size={logits.shape[2]})")
    print(f"Logits for first token (first 10 vocab items): {logits[0, 0, :10]}")
    print()

    # 步骤10: 预测下一个token
    next_token_logits = logits[0, -1, :]  # 最后一个位置的logits
    next_token_id = torch.argmax(next_token_logits).item()
    next_token = tokenizer.decode([next_token_id])
    print("步骤10: Next Token Prediction")
    print(f"Next token ID: {next_token_id}")
    print(f"Next token: '{next_token}'")
    print()

    # 步骤11: 概率分布
    probs = torch.softmax(next_token_logits, dim=-1)
    top_k = 5
    top_probs, top_indices = torch.topk(probs, top_k)
    top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
    print("步骤11: Top-5 Probabilities")
    for i in range(top_k):
        print(".4f")
    print()

print("=== 流程结束 ===")