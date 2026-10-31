<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/operations/gpu.md -->
<!-- i18n: source-sha256=ad37c7808c5ec6e92a59c5c365fcf8be49b9103acc62e9cf615b44a98e45bac3 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: 581e20f1

# GPU Operations

AnimaWorks can use the GPU for local embedding and reranker. Runtime device selection and failure status are handled by `core/infra/gpu.py`, and search components fall back to CPU when CUDA is unavailable or a runtime failure occurs.

## Device Configuration

The `GPUConfig` of `core/config/schemas.py` defines the device preference for embedding and reranker, the embedding batch size, and the number of batches that bulk processing yields to interactive processing. Devices are selected from `auto`, `cuda`, and `cpu`. See the [configuration reference](../reference/config.md) for items and default values.

The embedding `auto` also references the legacy `rag.use_gpu` configuration and selects GPU when CUDA is available and safe to use. The default device for reranker is CPU. Even if `cuda` is explicitly set, CPU is used when CUDA initialization fails.

```sh
animaworks config get gpu.embedding_device
animaworks config get gpu.reranker_device
animaworks config set gpu.embedding_device auto
```

## Behavior on Failure

When a GPU failure is detected during CUDA loading or encoding of the embedding model, the failure status is recorded and processing continues with the model on CPU. The reranker also retries on CPU on CUDA loading or inference errors. The status can be checked in the GPU information of the health/status response. Switching to CPU is a continuation strategy and does not repair driver or GPU hardware failures.

If it recurs, check the server log and the error in health/status. On the host side, the administrator investigates the recognition status of `nvidia-smi`, kernel and driver records, temperature, power, and connections. Do not assume GPU-required work until the cause of the failure is resolved and a safe restart is confirmed.

## Host-Side Failures

NVIDIA Xid 79 indicates that the GPU has been detached from the PCIe bus. Causes may include high load, power, connection, thermal, or driver issues, and the application switching to CPU does not resolve the host failure. If Xid 79 is followed by Xid 154 or similar, treat the GPU as unavailable and check the status after driver stack recovery or host restart.

Check power settings with `nvidia-smi -q -d POWER` and examine kernel records around the failure time with `journalctl -k --since "YYYY-MM-DD HH:MM"`. If it recurs, the administrator verifies power capacity and cables, PCIe connections, cooling, and driver stability. The repository includes `scripts/gpu-power-guard.sh` and systemd units, but the administrator applies only power limit values validated for the specific GPU and host. Do not apply examples intended for specific cards to other GPUs as-is.

## Performance Tuning

Reducing the embedding batch size helps lower peak memory but increases processing time. If large-scale index updates interfere with interactive processing, reduce bulk yield batches to give processing opportunities to interactive requests. After configuration changes, verify search quality and processing time with appropriate operational metrics rather than real data, and adjust along with GPU usage.
