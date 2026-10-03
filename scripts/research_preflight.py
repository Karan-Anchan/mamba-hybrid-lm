"""Inspect research runtime prerequisites without installing or training anything.

Run this with the Python environment intended for the experiment. Package metadata
does not prove a kernel works. Even --probe-torch only queries PyTorch and CUDA;
it never executes a Mamba kernel or certifies a backend for the research study.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any


PACKAGES = ("torch", "mamba-ssm", "triton", "causal-conv1d")
COMMAND_TIMEOUT_SECONDS = 5


def _decode(output: bytes) -> str:
    # wsl.exe may emit UTF-16 even when stdout is redirected.
    encoding = "utf-16-le" if b"\x00" in output else "utf-8"
    return output.decode(encoding, errors="replace").strip().lstrip("\ufeff")


def command_probe(name: str, arguments: list[str]) -> dict[str, Any]:
    executable = shutil.which(name)
    if executable is None:
        return {"status": "not_found", "executable": None}
    options: dict[str, Any] = {}
    if sys.platform == "win32":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        result = subprocess.run(
            [executable, *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
            **options,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "executable": executable,
            "timeout_seconds": COMMAND_TIMEOUT_SECONDS,
        }
    except OSError as error:
        return {"status": "error", "executable": executable, "error": str(error)}
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "executable": executable,
        "returncode": result.returncode,
        "stdout": _decode(result.stdout)[:8000],
        "stderr": _decode(result.stderr)[:2000],
    }


def package_versions() -> dict[str, dict[str, Any]]:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = {"installed": True, "version": metadata.version(name)}
        except metadata.PackageNotFoundError:
            packages[name] = {"installed": False, "version": None}
    return packages


def torch_probe(requested: bool) -> dict[str, Any]:
    if not requested:
        return {"status": "not_requested", "cuda_operation_executed": False}
    try:
        import torch

        report: dict[str, Any] = {
            "status": "ok",
            "version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_operation_executed": False,
            "devices": [],
        }
        if report["cuda_available"]:
            report["compiled_architecture_list"] = torch.cuda.get_arch_list()
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                with torch.cuda.device(index):
                    try:
                        # Exclude emulation: an emulation check can allocate test tensors.
                        bf16_supported = torch.cuda.is_bf16_supported(including_emulation=False)
                    except TypeError:
                        bf16_supported = None
                report["devices"].append({
                    "index": index,
                    "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory_bytes": properties.total_memory,
                    "bf16_supported_by_torch_query": bf16_supported,
                    "bf16_query_excludes_emulation": True,
                })
        return report
    except Exception as error:
        return {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "cuda_operation_executed": False,
        }


def collect_preflight(probe_torch: bool = False) -> dict[str, Any]:
    nvcc = command_probe("nvcc", ["--version"])
    nvidia = command_probe("nvidia-smi", ["--version"])
    gpu_inventory = command_probe(
        "nvidia-smi",
        ["--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader,nounits"],
    )
    toolkit_match = re.search(r"release\s+(\d+\.\d+)", nvcc.get("stdout", ""))
    driver_match = re.search(
        r"CUDA (?:UMD )?version\s*:\s*(\d+\.\d+)",
        nvidia.get("stdout", ""),
        flags=re.IGNORECASE,
    )
    wsl = (
        command_probe("wsl", ["--list", "--quiet"])
        if sys.platform == "win32"
        else {"status": "not_applicable", "executable": None}
    )
    recommendations = [
        "Use this report to select an environment; backend readiness is not certified.",
        "Pin dependencies and record exact backend, device, precision, and source identities for each new run.",
        "Prove Mamba forward, gradient, and recurrent-state parity before claiming CUDA acceleration.",
    ]
    if sys.platform == "win32":
        recommendations.append(
            "For upstream Mamba/Triton, evaluate a separate WSL/Linux environment; "
            "a wsl.exe entry alone does not prove an installed distribution or usable GPU."
        )
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "packages": package_versions(),
        "commands": {
            "nvcc": nvcc,
            "nvidia_smi_version": nvidia,
            "gpu_inventory": gpu_inventory,
            "wsl_distributions": wsl,
        },
        "cuda_versions": {
            "toolkit_nvcc_release": toolkit_match.group(1) if toolkit_match else None,
            "driver_reported_cuda_version": driver_match.group(1) if driver_match else None,
            "explanation": (
                "Driver CUDA reports driver compatibility; nvcc reports the local toolkit. "
                "PyTorch's bundled CUDA runtime is reported only by --probe-torch. "
                "These values describe different components and need not be identical."
            ),
        },
        "torch_probe": torch_probe(probe_torch),
        "backend_readiness": {
            "status": "not_certified",
            "mamba_kernel_executed": False,
            "explanation": "Installed packages, discovered tools, and CUDA metadata are not kernel correctness or speed evidence.",
        },
        "recommendations": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-torch", action="store_true", help="Import torch and query CUDA metadata; execute no model or CUDA compute operation.")
    parser.add_argument("--output", type=Path, help="Also save JSON to this explicit path; no files are written by default.")
    arguments = parser.parse_args()
    report = collect_preflight(arguments.probe_torch)
    serialized = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if arguments.output is not None:
        try:
            arguments.output.write_text(serialized, encoding="utf-8")
        except OSError as error:
            parser.error(f"cannot save preflight JSON: {error}")
    sys.stdout.write(serialized)


if __name__ == "__main__":
    main()
