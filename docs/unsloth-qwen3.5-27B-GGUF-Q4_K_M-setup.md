# llama.cpp on Jetson Orin — Setup Guide

Tested on: Jetson AGX Orin 32GB, JetPack 5/6, CUDA compute capability 8.7.

---

## 1. Build llama.cpp

Install dependencies and build with CUDA support:

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y

git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split

cp llama.cpp/build/bin/llama-* llama.cpp/
```

Verify CUDA is detected:

```bash
./llama.cpp/llama-cli --version
# Expected: ggml_cuda_init: found 1 CUDA devices: Device 0: Orin, compute capability 8.7
```

---

## 2. Download a model

Use the HuggingFace CLI. The example below downloads only the Q4_K_M quantization:

```bash
hf download unsloth/Qwen3.5-27B-GGUF \
    --include "*Q4_K_M*" \
    --local-dir ~/ai/ggufs
```

The file lands at `~/ai/ggufs/Qwen3.5-27B-Q4_K_M.gguf` (~15 GB).

---

## 3. Memory considerations on Jetson Orin

Jetson uses unified memory (CPU and GPU share the same physical RAM). Two issues arise:

**KV cache:** `--ctx-size` directly controls KV cache size. For a 27B model on 32 GB:

| ctx-size | KV cache (approx) | Safe? |
|---|---|---|
| 16384 | ~6 GB | No — OOM with model loaded |
| 8192 | ~3 GB | Yes |
| 4096 | ~1.5 GB | Yes |

**Memory fragmentation:** After extended uptime, `cudaMalloc` may fail to allocate the model weights contiguously even with plenty of free RAM (`lfb` in `tegrastats` will show small values). A reboot resolves this. If you cannot reboot, omit `-ngl` to fall back to CPU-only inference (~6–7 tok/s for 27B).

---

## 4. Run as a detached server

```bash
nohup ./llama.cpp/llama-server \
    -m ~/ai/ggufs/Qwen3.5-27B-Q4_K_M.gguf \
    --alias "unsloth/Qwen3.5-27B" \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --ctx-size 8192 \
    --port 8001 \
    -ngl 99 \
    > ~/ai/qwen3.5-27b-server.log 2>&1 &

echo "PID: $!"
```

Watch startup:

```bash
tail -f ~/ai/qwen3.5-27b-server.log
```

Ready when the log shows: `srv  server listening at http://127.0.0.1:8001`

To stop the server:

```bash
kill <PID>
```

---

## 5. Send a simple chat message

```bash
curl -s http://127.0.0.1:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "unsloth/Qwen3.5-27B",
        "messages": [{"role": "user", "content": "hi"}]
    }' | python3 -m json.tool
```

The response has two fields:
- `content` — the final answer
- `reasoning_content` — the model's internal thinking chain (Qwen3.5 is a reasoning model)

---

## 6. Tool call test script

Save as `test_tool_call.py` and run with `uv run test_tool_call.py`.

```python
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai"]
# ///

import json
import logging
import random
import subprocess

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# --- Tool implementations ---

def add_number(a, b):
    return float(a) + float(b)

def multiply_number(a, b):
    return float(a) * float(b)

def substract_number(a, b):
    return float(a) - float(b)

def write_a_story():
    return random.choice([
        "A long time ago in a galaxy far far away...",
        "There were 2 friends who loved sloths and code...",
        "The world was ending because every sloth evolved to have superhuman intelligence...",
        "Unbeknownst to one friend, the other accidentally coded a program to evolve sloths...",
    ])

def terminal(command):
    if any(w in command for w in ("rm", "sudo", "dd", "chmod")):
        msg = "Cannot execute dangerous commands"
        log.warning("Blocked command: %s", command)
        return msg
    log.info("Executing terminal command: %s", command)
    try:
        return subprocess.run(command, capture_output=True, text=True, shell=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed: {e.stderr}"

def python(code):
    log.info("Executing Python code:\n%s", code)
    data = {}
    exec(code, data)
    del data["__builtins__"]
    return str(data)

MAP_FN = {
    "add_number": add_number,
    "multiply_number": multiply_number,
    "substract_number": substract_number,
    "write_a_story": write_a_story,
    "terminal": terminal,
    "python": python,
}

# --- Tool schemas ---

tools = [
    {"type": "function", "function": {"name": "add_number", "description": "Add two numbers.", "parameters": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}}},
    {"type": "function", "function": {"name": "multiply_number", "description": "Multiply two numbers.", "parameters": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}}},
    {"type": "function", "function": {"name": "substract_number", "description": "Substract two numbers.", "parameters": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}}},
    {"type": "function", "function": {"name": "write_a_story", "description": "Writes a random story.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "terminal", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "python", "description": "Run Python code.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]

# --- Inference loop ---

def unsloth_inference(messages, temperature=1.0, top_p=0.95, top_k=20, min_p=0.00):
    messages = messages.copy()
    client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="sk-no-key-required")
    model_name = next(iter(client.models.list())).id
    log.info("Using model: %s", model_name)

    while True:
        log.info("Sending %d message(s) to model", len(messages))
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice="auto",
            extra_body={"top_k": top_k, "min_p": min_p},
        )
        choice = response.choices[0]
        tool_calls = choice.message.tool_calls or []
        content = choice.message.content or ""
        log.info("Model response — content: %r, tool_calls: %d", content, len(tool_calls))

        messages.append({"role": "assistant", "tool_calls": [tc.to_dict() for tc in tool_calls], "content": content})

        if not tool_calls:
            break

        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            log.info("Calling tool %r with args %s", tc.function.name, args)
            result = MAP_FN[tc.function.name](**args)
            log.info("Tool %r returned: %s", tc.function.name, result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": str(result)})

    return messages


if __name__ == "__main__":
    for query in [
        "What is 123 + 456?",
        "Multiply 7 by 8, then subtract 10 from the result.",
        "Write me a story.",
        "What files are in the current directory?",
    ]:
        log.info("=" * 60)
        log.info("USER QUERY: %s", query)
        result = unsloth_inference([{"role": "user", "content": query}])
        log.info("FINAL ANSWER: %s", result[-1].get("content"))
        log.info("=" * 60)
```

Expected output for the 4 test queries:

```
USER QUERY: What is 123 + 456?
  → calls add_number(a=123, b=456) → 579.0
  FINAL ANSWER: The sum of 123 and 456 is 579.

USER QUERY: Multiply 7 by 8, then subtract 10 from the result.
  → calls multiply_number(a=7, b=8) → 56.0
  → calls substract_number(a=56, b=10) → 46.0
  FINAL ANSWER: The result ... is 46.

USER QUERY: Write me a story.
  → calls write_a_story() → "A long time ago..."
  FINAL ANSWER: (model elaborates on the story seed)

USER QUERY: What files are in the current directory?
  → calls terminal(command="ls -la")
  FINAL ANSWER: (formatted file listing)
