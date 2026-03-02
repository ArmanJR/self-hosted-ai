self hosted ai

my setup resources, files and docs for self hosted ai services.

> current main device: nvidia jetson agx orin 32gb (aarch64, cuda 12.6).

resources

`docs/unsloth-qwen3.5-27B-GGUF-Q4_K_M-setup.md`
1. build from source with cuda
2. model download via hf download
3. jetson-specific memory caveats (kv cache sizing, fragmentation)
4. detached server launch + how to check/stop it
5. simple curl chat test
6. full tool call test script with expected output
