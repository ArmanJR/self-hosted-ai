# Qwen3.5-27B Claude Opus Reasoning Distilled — Jetson Orin Deployment

## Hardware

- **Device**: NVIDIA Jetson Orin (compute capability 8.7)
- **Unified Memory**: 30,696 MiB (shared CPU/GPU)
- **CPU Threads**: 12

## Model

Download from hf: https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/tree/main

| Property | Value |
|----------|-------|
| File | `Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_M.gguf` |
| Architecture | Qwen3.5 (hybrid SSM + SWA + Full Attention) |
| Parameters | 26.90 B |
| Quantization | Q4_K_M (4.92 BPW) |
| File size | 15.39 GiB |
| Training context | 262,144 tokens |
| Layers | 64 (16 full attention, rest SSM/SWA) |
| GQA | 24 heads / 4 KV heads, head dim 256 |

The hybrid architecture means only 1 in 4 layers uses full O(n) attention KV cache.
The remaining layers use SSM (O(1) state) or sliding window attention (O(window_size)),
making long contexts far cheaper than a standard transformer.

## Server Command

```bash
nohup ./llama.cpp/llama-server \
    -m ~/ggufs/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_M.gguf \
    --alias "Qwen3.5-27B-Claude-Opus-Distilled" \
    --port 8001 \
    -np 1 \
    -fa on \
    --no-mmap \
    -ctk q8_0 \
    -ctv q8_0 \
    --ctx-size 131072 \
    --temp 0.6 \
    --top-k 20 \
    --top-p 0.95 \
    --min-p 0.0 \
    --cache-reuse 256 \
    --reasoning-format deepseek \
    > ~/ai/qwen3.5-27b-distilled-server.log 2>&1 &
```

### Parameter Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| `-np 1` | 1 parallel slot | Single user; all memory goes to context depth instead of parallel slots |
| `-fa on` | Flash attention | More memory-efficient attention computation |
| `--no-mmap` | Disable memory mapping | Required on Jetson — mmap causes double memory occupation in unified memory (page cache + CUDA allocation) |
| `-ctk q8_0 -ctv q8_0` | Quantized KV cache | Halves KV cache memory (4,352 MiB vs ~8,704 MiB at f16) with negligible quality loss |
| `--ctx-size 131072` | 128K context | 16x the previous 8K; fits within VRAM budget with ~8 GiB headroom |
| `--temp 0.6` | Temperature | Matches model metadata recommendation; focused output for code tasks |
| `--top-k 20` | Top-K sampling | Model metadata recommendation |
| `--top-p 0.95` | Top-P sampling | Model metadata recommendation |
| `--min-p 0.0` | Min-P disabled | Filtering handled by top_k + top_p |
| `--cache-reuse 256` | Prompt cache reuse | Reuses cached KV from prior requests sharing a common prefix (>256 tokens); speeds up multi-turn chat |
| `--reasoning-format deepseek` | Reasoning extraction | Extracts `<think>` blocks into `reasoning_content` field for OpenAI-compatible clients |

### Why not `-ngl 99`

On Jetson's unified memory, explicitly pinning `-ngl 99` prevents the `--fit` mechanism from
auto-adjusting if memory is tight. Using the default (`-ngl auto`) lets llama.cpp verify that all
65 layers fit on GPU while retaining the ability to fall back. In practice, all 65/65 layers were
offloaded to GPU.

## Memory Breakdown

| Component | GPU (MiB) | CPU (MiB) |
|-----------|----------:|----------:|
| Model weights | 15,081.52 | 682.03 |
| KV cache (q8_0, 128K ctx) | 4,352.00 | — |
| Recurrent state (SSM) | 149.62 | — |
| Compute buffers | 495.00 | 276.02 |
| **Total used** | **~20,078** | **~958** |
| **Free (of 30,696)** | **~8,067** | — |

## Benchmark Results

**Date**: 2026-03-30
**llama.cpp build**: 8187 (feefb9283)

### Generation Performance

| Test | Prompt Tokens | Completion Tokens | Wall Time (s) | Gen (tok/s) | Prompt (tok/s) |
|------|:---:|:---:|:---:|:---:|:---:|
| Short Q&A | 21 | 165 | 23.64 | 7.26 | 32.84 |
| Code generation | 33 | 1,024 | 142.85 | 7.20 | 69.76 |
| Code review (547 tok input) | 547 | 2,048 | 289.54 | 7.14 | 208.28 |
| Algorithm design | 48 | 2,048 | 286.78 | 7.17 | 84.22 |

### Server-Side Timing Detail

```
Test 1 — Short Q&A:
  prompt eval:    639.54 ms /    21 tokens ( 30.45 ms/tok,  32.84 tok/s)
  generation:   22713.99 ms /   165 tokens (137.66 ms/tok,   7.26 tok/s)

Test 2 — Code generation:
  prompt eval:    473.03 ms /    33 tokens ( 14.33 ms/tok,  69.76 tok/s)
  generation:  142202.22 ms /  1024 tokens (138.87 ms/tok,   7.20 tok/s)

Test 3 — Code review:
  prompt eval:   2626.22 ms /   547 tokens (  4.80 ms/tok, 208.28 tok/s)
  generation:  286698.07 ms /  2048 tokens (139.99 ms/tok,   7.14 tok/s)

Test 4 — Algorithm design:
  prompt eval:    569.92 ms /    48 tokens ( 11.87 ms/tok,  84.22 tok/s)
  generation:  285787.61 ms /  2048 tokens (139.54 ms/tok,   7.17 tok/s)
```

### Key Findings

- **Generation speed is stable at ~7.2 tok/s** regardless of prompt size or task type.
  This is consistent with the memory-bandwidth-bound nature of autoregressive decoding on Orin.
- **Prompt ingestion scales with batch size**: 33 tok/s at 21 tokens, up to 208 tok/s at 547 tokens.
  A 10K-token codebase would be ingested in ~48 seconds.
- **Reasoning works correctly**: All 4 tests produced `<think>` blocks, properly extracted
  into the `reasoning_content` field via `--reasoning-format deepseek`.
- **128K context is usable**: The heaviest test used only 2,595 tokens. The full 128K window
  supports large codebases, long conversations, and extended reasoning chains.
- **~138 ms per output token**: The bottleneck is memory bandwidth (Orin's ~205 GB/s),
  not compute. This is the expected throughput for a 27B Q4_K_M model on this hardware.

## Troubleshooting Notes

### OOM with `-ngl 99` and `--ctx-size 0`

Setting `-ngl 99` (explicit) + `--ctx-size 0` (262K from model) causes `--fit` to abort because
it cannot reduce GPU layers when the user pinned them. The projected 24,759 MiB exceeds available
memory by ~1,300 MiB. Fix: omit `-ngl` (defaults to auto) or use an explicit `--ctx-size`.

### OOM with `--mmap` on external SSD

On Jetson's unified memory, memory-mapping a 16 GiB file from an external SSD causes the kernel
page cache and CUDA to both occupy physical memory for the same data. `--no-mmap` avoids this by
reading weights directly into CUDA buffers.

### `NvMapMemAllocInternalTagged: error 12`

Benign on Jetson. Appears during CUDA buffer allocation as a fallback path in the NvMap allocator.
Does not indicate failure by itself — only matters if followed by `cudaMalloc failed: out of memory`.
