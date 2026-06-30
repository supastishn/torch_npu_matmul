import os
import time
import urllib.request
import torch
import torch.nn as nn
import pytorch_npublas
dataset_path = "tinyshakespeare.txt"
if not os.path.exists(dataset_path):
    print("Downloading TinyShakespeare dataset...")
    try:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, dataset_path)
    except Exception as e:
        print(f"Download failed: {e}. Generating dummy Shakespeare text as fallback...")
        dummy_text = "To be, or not to be, that is the question.\n" * 1000
        with open(dataset_path, "w") as f:
            f.write(dummy_text)
with open(dataset_path, "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char_to_id[c] for c in s]
decode = lambda l: ''.join([id_to_char[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
class DSPLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    def forward(self, x):
        out = torch.matmul(x, self.weight.transpose(-2, -1))
        if self.bias is not None:
            out = out + self.bias
        return out
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = DSPLinear(d_model, d_model)
        self.k_proj = DSPLinear(d_model, d_model)
        self.v_proj = DSPLinear(d_model, d_model)
        self.out_proj = DSPLinear(d_model, d_model)
        self.mlp_l1 = DSPLinear(d_model, d_mlp)
        self.mlp_l2 = DSPLinear(d_mlp, d_model)
        self.activation = nn.GELU()
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att_weights = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att_weights = att_weights.masked_fill(causal_mask, float('-inf'))
        att_probs = torch.softmax(att_weights, dim=-1)
        att_out = torch.matmul(att_probs, v)
        att_out = att_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.out_proj(att_out)
        mlp_out = self.mlp_l2(self.activation(self.mlp_l1(x)))
        x = x + mlp_out
        return x
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, block_size=64, n_heads=4, d_mlp=256, num_layers=3):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = DSPLinear(d_model, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.pos_embedding(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)
        return logits, loss
def get_batch(data, batch_size=64, block_size=64):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y
def main():
    print("=== TINY SHAKESPEARE TRANSFORMER BENCHMARK ===")
    model = TinyTransformer(vocab_size=vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameter Count: {param_count / 1000:.2f}k parameters")
    x, y = get_batch(data, batch_size=64, block_size=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2)
    cpu_times = []
    print("\nRunning CPU training benchmarks...")
    for step in range(10):
        start = time.perf_counter()
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        elapsed = (time.perf_counter() - start) * 1000
        cpu_times.append(elapsed)
        print(f"CPU Step {step}: {elapsed:.2f} ms | Loss: {loss.item():.4f}")
    avg_cpu = sum(cpu_times) / len(cpu_times)
    print(f"Average CPU Step Time: {avg_cpu:.2f} ms")
    print("\nWarming up Hexagon CDSP backend...")
    old_matmul_patch = pytorch_npublas.matmul_patch
    def patched_with_dsp(input_a, input_b, *, out=None):
        return old_matmul_patch(input_a, input_b, out=out, backend_choice=2)
    torch.matmul = patched_with_dsp
    try:
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    except Exception as e:
        print(f"NPU Warmup warning/error: {e}")
    npu_times = []
    print("\nRunning Hexagon CDSP training benchmarks...")
    for step in range(10):
        start = time.perf_counter()
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        elapsed = (time.perf_counter() - start) * 1000
        npu_times.append(elapsed)
        print(f"CDSP Step {step}: {elapsed:.2f} ms | Loss: {loss.item():.4f}")
    avg_npu = sum(npu_times) / len(npu_times)
    print(f"\nAverage CPU Step Time: {avg_cpu:.2f} ms")
    print(f"Average CDSP Step Time: {avg_npu:.2f} ms")
    speedup = avg_cpu / avg_npu if avg_npu > 0 else 0.0
    print(f"CDSP vs CPU Speedup: {speedup:.2f}x")
if __name__ == "__main__":
    main()