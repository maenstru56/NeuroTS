from __future__ import annotations

from typing import Any

import torch


def resource_snapshot(tag: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"tag": str(tag)}
    try:
        import psutil

        process = psutil.Process()
        snapshot["cpu_percent"] = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        snapshot["ram_used_gb"] = (memory.total - memory.available) / (1024**3)
        snapshot["ram_total_gb"] = memory.total / (1024**3)
        snapshot["ram_percent"] = memory.percent
        snapshot["proc_rss_gb"] = process.memory_info().rss / (1024**3)
    except Exception:
        pass
    if torch.cuda.is_available():
        gpus = []
        for device_index in range(torch.cuda.device_count()):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                allocated_bytes = torch.cuda.memory_allocated(device_index)
                reserved_bytes = torch.cuda.memory_reserved(device_index)
                gpus.append(
                    {
                        "index": device_index,
                        "name": torch.cuda.get_device_name(device_index),
                        "allocated_gb": allocated_bytes / (1024**3),
                        "reserved_gb": reserved_bytes / (1024**3),
                        "used_gb": (total_bytes - free_bytes) / (1024**3),
                        "total_gb": total_bytes / (1024**3),
                    }
                )
            except Exception:
                continue
        snapshot["gpus"] = gpus
    return snapshot


def format_resource_snapshot(snapshot: dict[str, Any]) -> str:
    parts: list[str] = []
    if "cpu_percent" in snapshot:
        parts.append(f"cpu={float(snapshot['cpu_percent']):.1f}%")
    if "ram_used_gb" in snapshot and "ram_total_gb" in snapshot:
        parts.append(
            f"ram={float(snapshot['ram_used_gb']):.1f}/{float(snapshot['ram_total_gb']):.1f}GiB"
            f" ({float(snapshot.get('ram_percent', 0.0)):.1f}%)"
        )
    if "proc_rss_gb" in snapshot:
        parts.append(f"proc={float(snapshot['proc_rss_gb']):.1f}GiB")
    for gpu in snapshot.get("gpus", []):
        parts.append(
            f"gpu{int(gpu['index'])}=mem={float(gpu['used_gb']):.1f}/{float(gpu['total_gb']):.1f}GiB"
            f" torch={float(gpu['allocated_gb']):.1f}/{float(gpu['reserved_gb']):.1f}GiB"
        )
    for key, value in snapshot.items():
        if key in {"tag", "cpu_percent", "ram_used_gb", "ram_total_gb", "ram_percent", "proc_rss_gb", "gpus"}:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)
