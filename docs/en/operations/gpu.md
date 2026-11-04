<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/operations/gpu.md -->
<!-- i18n: source-sha256=d1884fcbfff860b73ade714750f2d4ff0ac207538925f825253f91797e19509f generated=2026-11-04 engine=local model=deepseek-v4.1-flash-1017-abl translator=2 -->

> Confirmed commit: 581e20f1

# GPU Operations

AnimaWorks can use a GPU (NVIDIA CUDA or Apple Silicon MPS) for local embedding and reranker. `core/infra/gpu.py` handles the selection of the execution device and failure status, and the search components switch to CPU when a GPU is unavailable or a failure occurs at runtime.

## Device Configuration

`GPUConfig` in `core/config/schemas.py` defines the desired device for embedding and reranker, the embedding batch size, and the number of batches that bulk processing yields to interactive processing. The device is chosen from `auto`, `cuda`, `mps`, and `cpu`. For the items and default values, see the [configuration reference](../reference/config.md).

`auto` selects whichever is available in the order CUDA, MPS, CPU. The embedding's `auto` also references the legacy `rag.use_gpu` setting, and uses CPU if it is disabled. The reranker's default device is CPU. If `cuda` or `mps` is explicitly specified but that device is unavailable, it is replaced with an available GPU, or CPU if none exists. The replacement is logged only once per component.

```sh
animaworks config get gpu.embedding_device
animaworks config get gpu.reranker_device
animaworks config set gpu.embedding_device auto
```

## Behavior on Failure

When a GPU failure (including MPS out-of-memory) is detected during loading of the embedding model onto the GPU or during encode, the failure status is recorded and processing continues with the model on CPU. The reranker likewise retries on CPU when a GPU load or inference error occurs. After switching, it remains on CPU until the server is restarted. The status can be checked in the GPU information of the health/status response. Switching to CPU is a measure to keep processing going and does not repair driver or GPU hardware failures.

If it recurs, check the server log and the health/status error. On the CUDA host side, an administrator investigates the recognition status of `nvidia-smi`, kernel and driver records, temperature, power, and connections. Until the cause of the failure is resolved and a safe restart is confirmed, do not assume that work requiring a GPU can be performed.

## Host-Side Failures

NVIDIA Xid 79 indicates that the GPU has been detached from the PCIe bus. Causes may include high load, power, connection, thermal, or driver issues, and the application switching to CPU does not resolve the host failure. If Xid 79 is followed by Xid 154 or similar, treat the GPU as unavailable and check the status after driver stack recovery or host restart.

Check power settings with `nvidia-smi -q -d POWER` and examine kernel records around the failure time with `journalctl -k --since "YYYY-MM-DD HH:MM"`. If it recurs, the administrator verifies power capacity and cables, PCIe connections, cooling, and driver stability. The repository includes `scripts/gpu-power-guard.sh` and systemd units, but the administrator applies only power limit values validated for the specific GPU and host. Do not apply examples intended for specific cards to other GPUs as-is.

## Performance Tuning

Reducing the embedding batch size helps lower peak memory but increases processing time. If large-scale index updates interfere with interactive processing, reduce bulk yield batches to give processing opportunities to interactive requests. After configuration changes, verify search quality and processing time with appropriate operational metrics rather than real data, and adjust along with GPU usage.

## Thread Count

torch models run both loading and inference on a dedicated thread in `core/infra/inference.py`. Because OpenMP creates a worker pool per thread that calls torch, running on the caller's thread increases the thread count in proportion to the number of callers. On CPU and CUDA, embedding and reranker run on separate threads, while on MPS both share a single thread. This is because torch's MPS does not support concurrent use from multiple threads, and calling it concurrently causes the process to terminate abnormally with a Metal assertion that Python cannot catch. New code that calls the model must always go through `run_inference`.

The torch CPU thread count is capped by `gpu.torch_num_threads` and `gpu.torch_num_interop_threads`. The tokio and rayon thread counts inside ChromaDB are capped by `rag.chroma_native_threads`. In all cases, `0` restores the library default (number of CPU cores), and a process restart is required for changes to take effect. If `TOKIO_WORKER_THREADS` and `RAYON_NUM_THREADS` are specified via environment variables, those values take precedence.

## Operating with MPS

Embedding on MPS and CPU is effectively identical, so there is no need to rebuild the existing vector index. `scripts/verify_mps_inference.py` is a script that verifies output agreement, concurrent execution, memory, and switching to CPU with the real model, and is not run in CI.

`gpu.embedding_batch_size` and `rag.embedding_max_seq_length` determine MPS memory usage. For a batch of 8 documents of 2048 tokens, the peak was about 3.1 GiB against about 2.1 GiB after model loading, and for 32 documents it was about 7.4 GiB. Released buffers are reused by torch, so there is no need to clear the cache per batch, and if memory runs out it switches to CPU. On Macs with little memory, reduce the batch size.
