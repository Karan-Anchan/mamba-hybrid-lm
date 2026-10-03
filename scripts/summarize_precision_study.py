"""Validate a declared precision matrix and export compact, uncertified study data.

Uses only the Python standard library. No model execution, checkpoint loading or
training takes place. Raw reports remain read-only and every failed probe remains
visible in the exported counts and check summaries.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TOLERANCES = {"float32": {"atol": 3e-5, "rtol": 3e-4}, "bfloat16": {"atol": 2e-3, "rtol": 2e-2}}
POLICIES = {
    "fp32_reference": {"id": "fp32_reference", "bf16": False, "scan_fp32": False, "projection_fp32": False},
    "bf16_original": {"id": "bf16_original", "bf16": True, "scan_fp32": False, "projection_fp32": False},
    "bf16_scan_fp32": {"id": "bf16_scan_fp32", "bf16": True, "scan_fp32": True, "projection_fp32": False},
    "bf16_projection_fp32": {"id": "bf16_projection_fp32", "bf16": True, "scan_fp32": False, "projection_fp32": True},
    "bf16_scan_and_projection_fp32": {"id": "bf16_scan_and_projection_fp32", "bf16": True, "scan_fp32": True, "projection_fp32": True},
}
SCAN_CHECKS = {"zero_one_shot_vs_full", "zero_bounded_vs_full", "zero_tokenwise_vs_full", "zero_final_state",
               "nonzero_bounded_output", "nonzero_tokenwise_output", "nonzero_bounded_state", "nonzero_tokenwise_state",
               "zero_output_vs_recurrence_oracle", "zero_state_vs_recurrence_oracle",
               "nonzero_output_vs_recurrence_oracle", "nonzero_state_vs_recurrence_oracle"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read(path: Path) -> tuple[dict, str]:
    def invalid(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), parse_constant=invalid)
    _require(isinstance(value, dict), f"JSON document must be an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _digest(value, label: str) -> None:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"invalid SHA256: {label}")


def _metric(value: dict, label: str) -> bool:
    _require(isinstance(value, dict) and type(value.get("finite")) is bool and type(value.get("passed")) is bool,
             f"invalid finite/pass flags: {label}")
    if value["finite"]:
        for key in ("max_absolute_error", "max_tolerance_ratio"):
            number = value.get(key)
            _require(type(number) in (int, float) and math.isfinite(number) and number >= 0, f"invalid metric: {label}.{key}")
        expected = value["max_tolerance_ratio"] <= 1
    else:
        _require(value.get("max_absolute_error") is None and value.get("max_tolerance_ratio") is None,
                 f"nonfinite metric must contain null errors: {label}")
        expected = False
    _require(value["passed"] == expected, f"inconsistent metric passed flag: {label}")
    return expected


def _metrics(values: dict, expected_keys: set, label: str) -> bool:
    _require(isinstance(values, dict) and set(values) == expected_keys, f"missing/extra comparison keys: {label}")
    passed = [_metric(value, f"{label}.{name}") for name, value in values.items()]
    return all(passed)


def _state(value: dict, fields: set, label: str) -> bool:
    _require(isinstance(value, dict) and isinstance(value.get("fields"), list), f"invalid retained state: {label}")
    names = [field.get("field") for field in value["fields"]]
    _require(len(names) == len(set(names)) and set(names) == fields, f"missing/duplicate retained-state fields: {label}")
    expected = all([_metric(field, f"{label}.{field['field']}") for field in value["fields"]])
    _require(type(value.get("passed")) is bool and value["passed"] == expected, f"inconsistent state passed flag: {label}")
    return expected


def _validate_case(case: dict, protocol: dict, inputs: dict, treatment: str) -> None:
    length, cfg = case["length"], protocol["model_config"]
    _require(case["precision_policy"] == POLICIES[treatment], "case precision policy differs from declaration")
    _require(case["weights_sha256"] == protocol["weights_sha256"], "case weights changed within a contrast")
    _require(all(case[key] == inputs[length][key] for key in ("tokens_sha256", "targets_sha256")), "case inputs changed within a contrast")
    tolerance = TOLERANCES["bfloat16" if POLICIES[treatment]["bf16"] else "float32"]
    _require(case["tolerance"] == tolerance, "case tolerance differs from the original gate")
    layer_types = cfg["layer_types"]
    mamba_layers = [index for index, layer in enumerate(layer_types) if layer == "mamba"]
    fields = {f"layer_{index}.{field}" for index, layer in enumerate(layer_types)
              for field in (("conv", "ssm") if layer == "mamba" else ("key", "value"))}
    checks = case["comparisons"]
    internal_keys = {"bounded_prefill_vs_full", "gate_prefill_vs_full", "token_decode_vs_full"}
    if length > 1:
        internal_keys |= {"gate_decode_vs_full", "causality"}
    logits_pass = _metrics(checks["internal_consistency"], internal_keys, "internal_consistency")
    _require(set(checks["internal_retained_state"]) == {"gate_final", "token_decode"}, "missing internal state comparisons")
    state_passes = [_state(value, fields, f"internal_state.{name}") for name, value in checks["internal_retained_state"].items()]
    _require(type(checks.get("internal_passed")) is bool and checks["internal_passed"] == (logits_pass and all(state_passes)),
             "inconsistent internal passed flag")
    anchor_pass = True
    if treatment == "fp32_reference":
        _require("fp32_anchor" not in checks, "FP32 reference must not claim an independent self-anchor comparison")
    else:
        anchor = checks["fp32_anchor"]
        anchor_keys = {"full", "bounded_prefill", "gate_prefill", "token_decode"} | ({"gate_decode"} if length > 1 else set())
        anchor_logits = _metrics(anchor["logits"], anchor_keys, "fp32_anchor.logits")
        _require(set(anchor["retained_state"]) == {"bounded_prefill", "gate_final", "token_decode"}, "missing anchor state comparisons")
        anchor_states = [_state(value, fields, f"anchor_state.{name}") for name, value in anchor["retained_state"].items()]
        anchor_pass = anchor_logits and all(anchor_states)
        _require(type(anchor.get("passed")) is bool and anchor["passed"] == anchor_pass, "inconsistent FP32-anchor passed flag")
    hidden_pass = _metrics(case["projection_input_consistency"], {"mixer_projection", "lm_projection"}, "projection_inputs")
    stage_rows = case["layer_stage_consistency"]
    expected_stages = [(layer, stage) for layer in mamba_layers for stage in ("scan_output", "gate_norm", "mixer_output")]
    _require([(row["layer"], row["stage"]) for row in stage_rows] == expected_stages, "missing/duplicate layer-stage observations")
    stage_passes = [_metric(row, f"layer_{row['layer']}.{row['stage']}") for row in stage_rows]
    first_failed = next(({"layer": row["layer"], "stage": row["stage"]} for row in stage_rows if not row["passed"]), None)
    _require(case["first_failed_layer_stage"] == first_failed, "first-divergence flag is inconsistent")
    _require(type(case.get("passed")) is bool and case["passed"] == (checks["internal_passed"] and anchor_pass and hidden_pass and all(stage_passes)),
             "inconsistent overall case passed flag")
    metadata = case["backend"]
    _require(metadata["requested"] == metadata["resolved"] == protocol["backend"] == "reference"
             and metadata["chunk_size"] == protocol["chunk_size"] and metadata["mamba_layers"] == len(mamba_layers)
             and metadata["fused"] is False, "backend metadata differs from this reference precision matrix")
    expected_paths = {"training": "torch.quadratic_ssd", "prefill": "torch.chunked_ssd", "decode": "torch.chunked_ssd"}
    _require(metadata["paths"] == expected_paths, "intended backend paths differ")
    stages = {"full", "bounded_prefill", "gate_prefill", "token_decode"} | ({"gate_decode", "causality"} if length > 1 else set())
    observed = defaultdict(set)
    _require(isinstance(case["operator_observations"], list) and bool(case["operator_observations"]), "missing actual operator observations")
    for event in case["operator_observations"]:
        stage = event["stage"]
        path = "torch.quadratic_ssd" if stage in ("full", "causality") else "torch.chunked_ssd"
        _require(stage in stages and event["layer"] in mamba_layers and event["path"] == path, "unexpected actual scan path")
        _require(type(event["calls"]) is int and event["calls"] > 0 and 0 < event["length"] <= length, "invalid operator call count/length")
        _require(event["autocast"] is (POLICIES[treatment]["bf16"] and not POLICIES[treatment]["scan_fp32"]), "actual scan autocast differs from treatment")
        _require(isinstance(event["input_dtypes"], list) and isinstance(event["output_dtypes"], list), "missing actual operator dtypes")
        observed[stage].add(path)
    _require(set(observed) == stages and case["observed_paths"] == {stage: sorted(paths) for stage, paths in observed.items()},
             "declared observed paths differ from operator observations")


def _validate_probe(probe: dict, length: int) -> None:
    scan = probe["scan"]
    _require(set(scan["operand_sha256"]) == {"x", "dt", "A", "B", "C", "D"}, "missing frozen scan operands")
    for name, value in scan["operand_sha256"].items():
        _digest(value, name)
    _digest(scan["initial_state_sha256"], "nonzero initial state")
    _require(set(probe["projections"]) == {"mixer_output", "lm_head"}, "unexpected projection isolation scope")
    for projection in probe["projections"].values():
        _digest(projection["input_sha256"], "projection input")
    groups = [("scan", scan), ("direct_gradients", scan["direct_gradients"]), *probe["projections"].items()]
    for name, group in groups:
        cases = group["cases"]
        _require([case["autocast"] for case in cases] == [True, False], "missing/duplicate exploratory precision contrasts")
        for case in cases:
            _require(case["tolerance"] == TOLERANCES["bfloat16" if case["autocast"] else "float32"], "probe tolerance changed")
            keys = (SCAN_CHECKS | ({"causality"} if length > 1 else set()) if name == "scan"
                    else {"x", "dt", "A", "B", "C", "D", "initial_state"} if name == "direct_gradients"
                    else {"bounded_vs_full", "tokenwise_vs_full"})
            passed = _metrics(case["checks"], keys, f"probe.{name}")
            if name == "scan":
                _require(case["contraction_inputs_dtype"] == "torch.float32"
                         and case["cb_contraction_output_dtype"] == ("torch.bfloat16" if case["autocast"] else "torch.float32")
                         and case["retained_state_dtype"] == "torch.float32", "unexpected observed scan precision")
            if name == "direct_gradients":
                future = case["future_gradient_causality"]
                _require((future is None) == (length == 1), "missing future-gradient causality probe")
                if future is not None:
                    _require([item["input"] for item in future["inputs"]] == ["x", "dt", "B", "C"], "missing future-gradient inputs")
                    for item in future["inputs"]:
                        gradient = item["max_future_absolute_gradient"]
                        _require(type(item["finite"]) is bool and type(item["passed"]) is bool
                                 and type(gradient) in (int, float) and math.isfinite(gradient) and gradient >= 0
                                 and item["passed"] == (item["finite"] and gradient == 0), "inconsistent future-gradient check")
                    _require(future["passed"] == all(item["passed"] for item in future["inputs"]), "inconsistent future-gradient causality flag")
                    passed = passed and future["passed"]
            _require(type(case["passed"]) is bool and case["passed"] == passed, "inconsistent exploratory probe passed flag")


def _validate_report(report: dict, declaration: dict) -> None:
    _require(report["schema"] == 1 and report["certified"] is False and report["execution_status"] == "completed", "report is incomplete or incorrectly certified")
    protocol = report["protocol"]
    expected_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    _require(report["protocol_sha256"] == expected_hash, "raw report protocol hash does not match its contents")
    _require(protocol["version"] == 1 and protocol["seed"] in declaration["seeds"]
             and protocol["model_config"]["ratio"] in declaration["ratios"], "unexpected raw report matrix arm")
    _require(protocol["device"] == "cuda" and protocol["backend"] == "reference" and protocol["checkpoint"] is None
             and protocol["batch_size"] == 2 and protocol["chunk_size"] == declaration["chunk_size"], "raw report workload differs from declaration")
    _require(protocol["tolerances"] == declaration["tolerances"] == TOLERANCES, "original tolerances changed")
    cfg = protocol["model_config"]
    _require(all(cfg[key] == value for key, value in declaration["model_geometry"].items()), "model geometry differs from declaration")
    a, s = (int(value) for value in cfg["ratio"].split(":"))
    _require(cfg["layer_types"] == ["attention" if index % (a + s) >= s else "mamba" for index in range(cfg["n_layers"])], "model placement differs from declared ratio")
    treatments = declaration["treatments"] + declaration["exploratory_treatments"]
    _require(protocol["treatments"] == [POLICIES[name] for name in treatments], "missing/reordered treatment policy")
    _digest(protocol["weights_sha256"], "model weights")
    inputs = protocol["inputs"]
    _require([item["length"] for item in inputs] == declaration["lengths"], "input lengths differ from declared matrix")
    for item in inputs:
        _digest(item["tokens_sha256"], "tokens")
        _digest(item["targets_sha256"], "targets")
    input_map = {item["length"]: item for item in inputs}
    expected_cases = [(name, length) for name in treatments for length in declaration["lengths"]]
    _require([(case["treatment"], case["length"]) for case in report["cases"]] == expected_cases, "missing/duplicate/reordered report cases")
    for case in report["cases"]:
        _validate_case(case, protocol, input_map, case["treatment"])
    _require([probe["length"] for probe in report["exploratory_probes"]] == declaration["lengths"], "missing/duplicate exploratory probes")
    for probe in report["exploratory_probes"]:
        _validate_probe(probe, probe["length"])
    for path, digest in protocol["source_sha256"].items():
        _digest(digest, f"source {path}")
    _require("scripts/study_scan_precision.py" in protocol["source_sha256"], "study source fingerprint missing")
    runtime = protocol["runtime"]
    _require(report["packages"] == runtime["packages"] and report["python"] == runtime["python"]
             and report["torch_cuda_runtime"] == runtime["torch_cuda_runtime"] and report["precision_flags"] == runtime["precision_flags"]
             and report["gpu"] == runtime["gpu"], "runtime fields differ from signed protocol")
    all_pass = all(case["passed"] for case in report["cases"]) and all(case["passed"] for probe in report["exploratory_probes"]
                  for group in [probe["scan"], probe["scan"]["direct_gradients"], *probe["projections"].values()] for case in group["cases"])
    _require(report["status"] == ("completed" if all_pass else "completed_with_parity_failures"), "report status hides or invents parity failures")


def _count(values: list[bool]) -> dict:
    return {"passed": sum(values), "total": len(values)}


def _metric_summary(values: list[dict]) -> dict:
    finite = [value for value in values if value["finite"]]
    return {**_count([value["passed"] for value in values]), "nonfinite": len(values) - len(finite),
            "worst_tolerance_ratio": max((value["max_tolerance_ratio"] for value in finite), default=None),
            "worst_absolute_error": max((value["max_absolute_error"] for value in finite), default=None)}


def _group_summary(cases: list[dict]) -> dict:
    names = sorted({name for case in cases for name in case["checks"]})
    return {**_count([case["passed"] for case in cases]),
            "checks": {name: _metric_summary([case["checks"][name] for case in cases if name in case["checks"]]) for name in names}}


def build_summary(declaration_path: Path, report_paths: list[Path], repository_root: Path = ROOT) -> dict:
    root = Path(repository_root).resolve()
    declaration_path = Path(declaration_path).resolve()
    declaration, declaration_sha = _read(declaration_path)
    _require(declaration["schema"] == 1 and declaration["status_at_declaration"] == "planned_before_execution", "invalid declared protocol")
    treatments = declaration["treatments"] + declaration["exploratory_treatments"]
    _require(treatments == list(POLICIES) and declaration["tolerances"] == TOLERANCES, "declared treatments/tolerances differ from the original study")
    for name in ("seeds", "ratios", "lengths"):
        _require(bool(declaration[name]) and len(declaration[name]) == len(set(declaration[name])), f"duplicate/empty declaration {name}")
    expected = {(ratio, seed) for ratio in declaration["ratios"] for seed in declaration["seeds"]}
    _require(len(report_paths) == len(expected), "missing/extra raw reports in declared matrix")
    reports, registry, identities = [], [], set()
    for report_path in report_paths:
        path = Path(report_path).resolve()
        _require(path.is_relative_to(root), "raw report must remain within the exported repository")
        report, digest = _read(path)
        try:
            _validate_report(report, declaration)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"incomplete/malformed precision report {path.name}: {exc}") from exc
        protocol = report["protocol"]
        identity = (protocol["model_config"]["ratio"], protocol["seed"])
        _require(identity not in identities, "duplicate raw matrix arm")
        identities.add(identity)
        reports.append(report)
        registry.append({"path": path.relative_to(root).as_posix(), "sha256": digest,
                         "canonical_sha256": _canonical_hash(report), "protocol_sha256": report["protocol_sha256"],
                         "ratio": identity[0], "seed": identity[1], "cases": len(report["cases"]),
                         "weights_sha256": protocol["weights_sha256"], "status": report["status"]})
    _require(identities == expected, "missing/unexpected raw matrix arms")
    reports.sort(key=lambda report: (declaration["ratios"].index(report["protocol"]["model_config"]["ratio"]),
                                    declaration["seeds"].index(report["protocol"]["seed"])))
    first = reports[0]["protocol"]
    geometry = {key: value for key, value in first["model_config"].items() if key not in ("name", "ratio", "layer_types")}
    for report in reports:
        protocol = report["protocol"]
        _require(protocol["source_sha256"] == first["source_sha256"], "raw reports use inconsistent source versions")
        _require(protocol["runtime"] == first["runtime"], "raw reports use inconsistent precision/runtime settings")
        _require({key: value for key, value in protocol["model_config"].items() if key not in ("name", "ratio", "layer_types")} == geometry,
                 "raw reports use inconsistent model geometry")
    rows = []
    for ratio in declaration["ratios"]:
        for treatment in treatments:
            selected = [(report["protocol"]["seed"], case) for report in reports if report["protocol"]["model_config"]["ratio"] == ratio
                        for case in report["cases"] if case["treatment"] == treatment]
            cases = [case for _, case in selected]
            internal = _count([case["comparisons"]["internal_passed"] for case in cases])
            anchor = None if treatment == "fp32_reference" else _count([case["comparisons"]["fp32_anchor"]["passed"] for case in cases])
            logits = {name: _metric_summary([case["comparisons"]["internal_consistency"][name] for case in cases
                                            if name in case["comparisons"]["internal_consistency"]])
                      for name in sorted({name for case in cases for name in case["comparisons"]["internal_consistency"]})}
            failures = [{"seed": seed, "length": case["length"],
                         "internal_failed_checks": [name for name, check in case["comparisons"]["internal_consistency"].items() if not check["passed"]],
                         "fp32_anchor_passed": case["comparisons"].get("fp32_anchor", {}).get("passed"),
                         "first_failed_layer_stage": case["first_failed_layer_stage"]}
                        for seed, case in selected if not case["passed"]]
            rows.append({"ratio": ratio, "treatment": treatment, "exploratory": treatment in declaration["exploratory_treatments"],
                         "cases": len(cases), "internal": internal, "fp32_anchor": anchor, "logits": logits,
                         "retained_state": _count([all(state["passed"] for state in case["comparisons"]["internal_retained_state"].values()) for case in cases]),
                         "diagnostic_all": _count([case["passed"] for case in cases]), "failures": failures,
                         "by_seed": [{"seed": seed, "cases": len(values := [case for current, case in selected if current == seed]),
                                      "internal_passed": sum(case["comparisons"]["internal_passed"] for case in values),
                                      "fp32_anchor_passed": None if anchor is None else sum(case["comparisons"]["fp32_anchor"]["passed"] for case in values)}
                                     for seed in declaration["seeds"]]})
    probes = []
    for ratio in declaration["ratios"]:
        selected = [probe for report in reports if report["protocol"]["model_config"]["ratio"] == ratio for probe in report["exploratory_probes"]]
        for autocast in (True, False):
            groups = {"scan": [], "direct_gradients": [], "mixer_projection": [], "lm_projection": []}
            for probe in selected:
                for name, group in (("scan", probe["scan"]), ("direct_gradients", probe["scan"]["direct_gradients"]),
                                    ("mixer_projection", probe["projections"]["mixer_output"]), ("lm_projection", probe["projections"]["lm_head"])):
                    groups[name].append(next(case for case in group["cases"] if case["autocast"] == autocast))
            future = [case["future_gradient_causality"] for case in groups["direct_gradients"] if case["future_gradient_causality"] is not None]
            probes.append({"ratio": ratio, "autocast": autocast, "precision_label": "BF16 autocast" if autocast else "autocast disabled",
                           "cases": len(selected), **{name: _group_summary(cases) for name, cases in groups.items()},
                           "future_gradient_causality": {**_count([case["passed"] for case in future]), "not_applicable": len(selected) - len(future)},
                           "cb_contraction_output_dtypes": sorted({case["cb_contraction_output_dtype"] for case in groups["scan"]})})
    return {"schema": 1, "date": declaration["date"], "status": "completed_with_parity_failures" if any(report["status"] == "completed_with_parity_failures" for report in reports) else "completed",
            "certified": False, "scope": declaration["scope"],
            "declaration": {"path": declaration_path.relative_to(root).as_posix(), "sha256": declaration_sha,
                            "canonical_sha256": _canonical_hash(declaration)},
            "expected_matrix": {"ratios": declaration["ratios"], "seeds": declaration["seeds"], "lengths": declaration["lengths"],
                                "treatments": treatments, "raw_reports": len(expected), "cases_per_report": len(treatments) * len(declaration["lengths"]),
                                "cases_per_row": len(declaration["seeds"]) * len(declaration["lengths"])},
            "tolerances": TOLERANCES, "source_sha256": first["source_sha256"], "runtime": first["runtime"],
            "rows": rows, "exploratory_probes": probes,
            "raw_reports": sorted(registry, key=lambda row: (declaration["ratios"].index(row["ratio"]), row["seed"])),
            "limits": ["Counts describe fixed software diagnostic cases, not language quality or research significance.",
                       "Each main row contains seed × prompt-length cases; all cases reuse one model per seed.",
                       "Internal agreement and agreement with the common FP32 anchor remain separate.",
                       "Projection treatment covers Mamba out_proj and the LM head; attention/MLP projections remain unchanged.",
                       "Causality probes perturb or differentiate only the final token; they do not exhaust every possible future position.",
                       "Combined scan-and-projection precision changes are exploratory and do not isolate a single operation.",
                       "Hybrid/pure initialization consumes RNG differently; shared seeds do not show that attention stabilizes arithmetic.",
                       "Random tiny models, listed lengths and synthetic initial states do not certify production checkpoints or fused kernels.",
                       "Public data are generated from validated reports; failures and exploratory probe outcomes remain visible."]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _require(args.output.resolve() not in {args.protocol.resolve(), *(path.resolve() for path in args.reports)},
             "summary output must not overwrite its declaration or raw reports")
    summary = build_summary(args.protocol, args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": summary["status"], "certified": False, "rows": len(summary["rows"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
