# GPU Operations

This page covers the operational guardrails for AnimaWorks hosts that use an
NVIDIA GPU for local embedding or retrieval workloads.

## Xid 79 Context

NVIDIA Xid 79 means the driver reported that the GPU fell off the PCIe bus. On
high-power cards such as the RTX 3090, this can happen under sustained load with
short power spikes. A nightly consolidation run can create that pattern when
embedding work overlaps with other model activity.

When Xid 79 is followed by Xid 154, treat the GPU as unavailable until the host
has been restarted or the driver stack has been recovered. Application-level CPU
fallback can keep work moving, but it does not fix the underlying hardware or
driver state.

## Power Guard Setup

Use `scripts/gpu-power-guard.sh` to enable NVIDIA persistence mode and apply a
conservative power limit. The default limit is `280W`, which is intended for an
RTX 3090 class system where the stock limit may be around `350W`.

Run it manually:

```bash
sudo GPU_POWER_LIMIT_W=280 scripts/gpu-power-guard.sh
```

Install it for boot-time execution:

```bash
sudo install -m 0755 scripts/gpu-power-guard.sh /usr/local/bin/gpu-power-guard.sh
sudo install -m 0644 scripts/systemd/gpu-power-guard.service /etc/systemd/system/gpu-power-guard.service
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-power-guard.service
```

The script is safe on non-GPU systems. If `nvidia-smi` is missing, no NVIDIA GPU
is present, or an `nvidia-smi` command fails, it prints a warning and exits with
status `0` only when no GPU action is possible. If a GPU is present but the power
limit cannot be applied or verified, the script exits non-zero so systemd reports
the guard as failed.

To change the limit, edit the unit environment value or create a systemd drop-in:

```bash
sudo systemctl edit gpu-power-guard.service
```

```ini
[Service]
Environment=GPU_POWER_LIMIT_W=260
```

Then reload and restart the unit:

```bash
sudo systemctl daemon-reload
sudo systemctl restart gpu-power-guard.service
```

## Recurrence Checklist

If Xid 79 recurs after the power guard is installed:

- Confirm the configured power limit with `nvidia-smi -q -d POWER`.
- Check kernel logs around the failure time with `journalctl -k --since "YYYY-MM-DD HH:MM"`.
- Inspect PSU capacity, cabling, and dedicated PCIe power connectors for the GPU.
- Reseat the GPU and check the PCIe slot for link errors or instability.
- Disable PCIe ASPM in firmware or with the kernel command line if link power
  management is suspected.
- Review cooling and chassis airflow during the failing workload window.
- If failures continue at a lower power limit, treat the issue as hardware,
  power-delivery, motherboard, or driver instability rather than only an
  application load problem.
