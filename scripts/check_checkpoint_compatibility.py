"""Read-only short FP32 compatibility check on registered historical weights.

This verifies loading, finite outputs and same-weight short-context agreement.
It starts no optimizer and cannot replace held-out quality or long-context tests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
from src.eval.suite import discover_checkpoints, load_variant_model, evaluation_source_hashes  # noqa: E402
from scripts.check_scan_backend import compare  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--training-run-id", default="week3-700m-v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tolerance = {"atol": 3e-4, "rtol": 3e-4}
    checkpoints = discover_checkpoints(args.checkpoint_root, args.training_run_id)
    rows = []
    for ratio, checkpoint in checkpoints.items():
        model = load_variant_model(checkpoint, torch.device(args.device))
        torch.manual_seed(1337)
        tokens = torch.randint(0, model.cfg.vocab_size, (1, 17), device=args.device)
        with torch.no_grad():
            reference, _ = model(tokens)
            cached, state = model.prefill(tokens[:, :14])
            prefill = compare(cached, reference[:, 13:14], tolerance)
            decode = []
            for index in range(14, 17):
                decode.append(compare(model.decode(tokens[:, index:index + 1], state),
                                      reference[:, index:index + 1], tolerance))
            keys = tuple(model.state_dict())
            metadata = model.configure_scan_backend("torch_chunked", 8)
            bounded, _ = model(tokens)
            chunked = compare(bounded, reference, tolerance)
        rows.append({"ratio": ratio, "checkpoint_sha256": checkpoint.checkpoint_sha256,
                     "training_signature": checkpoint.signature,
                     "strict_load": True, "parameter_count": model.num_params(),
                     "state_dict_keys_unchanged": keys == tuple(model.state_dict()),
                     "reference_prefill": prefill, "reference_tokenwise_decode": decode,
                     "chunked_forward": chunked, "backend": metadata,
                     "passed": prefill["passed"] and all(d["passed"] for d in decode)
                               and chunked["passed"] and keys == tuple(model.state_dict())})
        del model, state, tokens, reference, bounded, cached
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    root = Path(__file__).resolve().parents[1]
    hashes = evaluation_source_hashes(root)
    hashes["scripts/check_scan_backend.py"] = hashlib.sha256((root / "scripts/check_scan_backend.py").read_bytes()).hexdigest()
    hashes["scripts/check_checkpoint_compatibility.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"schema": 1, "status": "passed" if all(r["passed"] for r in rows) else "failed",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": 1337,
              "device": args.device, "dtype": "float32", "length": 17, "tolerances": tolerance,
              "training_run_id": args.training_run_id, "torch": torch.__version__,
              "source_sha256": hashes, "cases": rows,
              "limits": "Short FP32 compatibility only; no new training, quality evaluation or fused kernel."}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
