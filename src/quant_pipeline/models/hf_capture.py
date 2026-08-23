from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from pathlib import Path

from ..calibration.windows import verify_sealed_corpus
from ..core.artifacts import bind_files, prepare_empty_destination, require_execute, sha256_file, write_json


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    values = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ",".join(values) or None


def capture_logits(
    model_path: str,
    sealed_corpus: str | Path,
    role: str,
    output_dir: str | Path,
    execute: bool,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    model_revision: str = "",
) -> dict:
    require_execute(execute, "load the model and capture logits")
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ValueError("capture requires the expected 40-hex model revision")
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM
    except Exception as error:  # pragma: no cover
        raise RuntimeError("install quant-pipeline[hf] for capture") from error
    sealed = json.loads(Path(sealed_corpus).read_text())
    verify_sealed_corpus(sealed)
    windows = sealed["windows"][role]
    destination = prepare_empty_destination(output_dir)
    torch_dtype = getattr(torch, dtype)
    local_model = Path(model_path)
    if not local_model.is_dir():
        raise ValueError("capture requires a local immutable checkpoint directory")
    config_path = local_model / "config.json"
    index_path = local_model / "model.safetensors.index.json"
    identity_files = [config_path]
    if index_path.exists():
        index = json.loads(index_path.read_text())
        identity_files.append(index_path)
        identity_files.extend(local_model / name for name in sorted(set(index["weight_map"].values())))
    elif (local_model / "model.safetensors").exists():
        identity_files.append(local_model / "model.safetensors")
    else:
        raise FileNotFoundError("checkpoint has no safetensors index or model.safetensors")
    model_identity = {"expected_revision": model_revision, "files": bind_files(identity_files)}
    model = AutoModelForCausalLM.from_pretrained(
        local_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.output_router_logits = True
    records = []
    input_device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        for index, window in enumerate(windows):
            input_ids = torch.tensor([window["token_ids"]], dtype=torch.long, device=input_device)
            output = model(input_ids=input_ids, use_cache=False, output_router_logits=True, return_dict=True)
            tensors = {"logits": output.logits[0, :-1].to(torch.float32).cpu()}
            for layer, router in enumerate(output.router_logits or ()):
                tensors[f"router_logits.layer_{layer:03d}"] = router.detach().to(torch.float32).cpu()
            path = destination / f"window-{index:04d}.safetensors"
            save_file(tensors, path, metadata={"token_sha256": window["token_sha256"], "role": role})
            records.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "token_sha256": window["token_sha256"]})
    receipt = {
        "schema": "quant-pipeline.hf-logit-capture.v1",
        "model_path": str(Path(model_path).resolve()) if Path(model_path).exists() else model_path,
        "model_identity": model_identity,
        "sealed_corpus_sha256": sha256_file(sealed_corpus),
        "role": role,
        "dtype": dtype,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nvidia_driver": _nvidia_driver_version(),
            "image_digest": os.environ.get("QUANT_PIPELINE_IMAGE_DIGEST"),
            "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
            "transformers": __import__("transformers").__version__,
            "safetensors": __import__("safetensors").__version__,
            "device_map": device_map,
            "model_class": type(model).__name__,
            "cuda_devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        },
        "records": records,
    }
    write_json(destination / "capture-receipt.json", receipt)
    return receipt
