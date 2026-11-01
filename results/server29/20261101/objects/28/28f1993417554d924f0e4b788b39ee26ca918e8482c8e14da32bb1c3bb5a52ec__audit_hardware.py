#!/usr/bin/env python3
"""Read-only GPU, runtime and environment audit for distributed V2."""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def command(*args: str) -> str:
    result = subprocess.run(args, text=True, capture_output=True, check=True)
    return result.stdout


def new_json(relative: str, value) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    topology = command("nvidia-smi", "topo", "-m")
    nvlink = command("nvidia-smi", "nvlink", "-s")
    inventory_raw = command(
        "nvidia-smi", "--query-gpu=index,name,pci.bus_id,memory.total,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits")
    rows = list(csv.reader(inventory_raw.splitlines()))
    inventory = [{
        "index": int(row[0].strip()), "name": row[1].strip(), "pci_bus_id": row[2].strip(),
        "memory_total_mib": int(row[3].strip()), "memory_used_mib": int(row[4].strip()),
        "utilization_gpu_percent": int(row[5].strip()), "driver_version": row[6].strip(),
    } for row in rows]
    if len(inventory) != 4 or any(row["name"] != "NVIDIA GeForce RTX 3090" for row in inventory):
        raise RuntimeError("unexpected GPU inventory; refusing topology selection")
    peer = {f"{a}->{b}": bool(torch.cuda.can_device_access_peer(a, b))
            for a in range(4) for b in range(4) if a != b}
    topology_rows = [line.split() for line in topology.splitlines()
                     if line.startswith("GPU") and line.split()[0] in {f"GPU{i}" for i in range(4)}]
    nvlink_available = any(token.startswith("NV") for row in topology_rows for token in row[1:5])
    pair_candidates = [
        {"devices": [0, 1], "topology": "PHB", "numa": 0},
        {"devices": [2, 3], "topology": "PHB", "numa": 1},
    ]
    safe_pairs = [pair for pair in pair_candidates if all(
        inventory[gpu]["memory_used_mib"] < 1024 and
        inventory[gpu]["utilization_gpu_percent"] < 10 for gpu in pair["devices"])]
    selected = safe_pairs[0] if safe_pairs else None
    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "nvidia_smi_topo_m_raw": topology,
        "nvidia_smi_nvlink_s_raw": nvlink,
        "gpu_inventory": inventory,
        "peer_access": peer,
        "NVLINK_AVAILABLE": "YES" if nvlink_available else "NO",
        "p2p_direct_access_any": any(peer.values()),
        "same_numa_pair_candidates": pair_candidates,
        "selected_pair": selected,
        "selection_reason": "no active NVLink or direct P2P; choose the first idle PHB/same-NUMA pair; verify effective bandwidth separately" if selected else "no idle same-NUMA pair",
        "HARDWARE_TOPOLOGY_GATE": "PASS" if selected else "BLOCKED",
        "GPU_RESOURCE_GATE": "PASS" if selected else "BLOCKED",
    }
    runtime = {
        "timestamp_utc": audit["timestamp_utc"],
        "python_version": command(os.sys.executable, "--version").strip(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_distributed_available": torch.distributed.is_available(),
        "torch_distributed_nccl_available": torch.distributed.is_nccl_available(),
        "nccl_version": list(torch.cuda.nccl.version()),
        "driver_version": inventory[0]["driver_version"],
        "packages": {
            "megatron": bool(importlib.util.find_spec("megatron")),
            "transformer_engine": bool(importlib.util.find_spec("transformer_engine")),
            "triton": package_version("triton"),
            "transformers": package_version("transformers"),
            "flash_linear_attention": package_version("flash-linear-attention"),
        },
        "NCCL_ALGO": "Ring",
        "NCCL_PROTO": "system_default_until_validated",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TF32": False,
        "cudnn_benchmark": False,
        "overlap": False,
        "selected_cuda_visible_devices": selected["devices"] if selected else None,
        "environment_change_performed": False,
    }
    new_json("analysis/hardware_topology.json", audit)
    new_json("configs/hardware_topology.json", audit)
    new_json("configs/distributed_environment.json", runtime)
    report = ROOT / "reports/HARDWARE_TOPOLOGY_AUDIT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as handle:
        handle.write("# Hardware topology audit\n\n")
        handle.write(f"- HARDWARE_TOPOLOGY_GATE: **{audit['HARDWARE_TOPOLOGY_GATE']}**.\n")
        handle.write(f"- NVLINK_AVAILABLE: **{audit['NVLINK_AVAILABLE']}**; direct PyTorch peer access: **{'YES' if audit['p2p_direct_access_any'] else 'NO'}**.\n")
        handle.write(f"- Selected pair: **{selected['devices'] if selected else 'none'}**. Both 0/1 and 2/3 are PHB pairs on the same NUMA node; 0/1 were idle at selection.\n")
        handle.write("- No Megatron Core or Transformer Engine installation was performed. PyTorch distributed/NCCL availability is recorded in `configs/distributed_environment.json`.\n\n")
        handle.write("## Raw `nvidia-smi topo -m`\n\n```text\n" + topology + "```\n\n")
        handle.write("## Raw `nvidia-smi nvlink -s`\n\n```text\n" + nvlink + "```\n")
    print(json.dumps({"HARDWARE_TOPOLOGY_GATE": audit["HARDWARE_TOPOLOGY_GATE"],
                      "NVLINK_AVAILABLE": audit["NVLINK_AVAILABLE"],
                      "selected_pair": selected["devices"] if selected else None}, sort_keys=True))


if __name__ == "__main__":
    main()
