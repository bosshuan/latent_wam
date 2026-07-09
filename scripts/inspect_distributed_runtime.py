"""Report the PyTorch FSDP/DCP capabilities required by production training."""

from __future__ import annotations

import json

import torch

from train.fsdp_checkpoint import dcp_capabilities


def main() -> None:
    from torch.distributed.fsdp import ShardingStrategy

    report = dcp_capabilities()
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_device_count"] = torch.cuda.device_count()
    report["full_shard"] = hasattr(ShardingStrategy, "FULL_SHARD")
    report["hybrid_shard"] = hasattr(ShardingStrategy, "HYBRID_SHARD")
    print("[dist-runtime] " + json.dumps(report, sort_keys=True))
    required = (
        "dcp_save",
        "dcp_load",
        "get_state_dict",
        "set_state_dict",
        "legacy_optimizer_load",
        "full_shard",
        "hybrid_shard",
    )
    missing = [name for name in required if not report[name]]
    if missing:
        raise RuntimeError(f"distributed runtime missing required capabilities: {missing}")
    print("[dist-runtime] ok", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
