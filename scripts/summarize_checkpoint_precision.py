"""Validate and summarize the separate trained-checkpoint K1 diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_precision_study import (
    POLICIES, TOLERANCES, _canonical_hash, _count, _digest, _group_summary,
    _read, _require, _validate_case, _validate_probe,
)

MEASURED_SOURCES = {
    'src/model/attention.py', 'src/model/block.py', 'src/model/config.py',
    'src/model/inference.py', 'src/model/lm.py', 'src/model/mamba2.py',
    'src/model/mlp.py', 'src/model/norm.py', 'src/model/scan_backend.py',
    'scripts/study_scan_precision.py', 'scripts/check_scan_backend.py',
}


def validate_sources(sources: dict, source_root: Path) -> None:
    """Require every measured runner/model/checker file and verify its fingerprint."""
    _require(isinstance(sources, dict) and set(sources) == MEASURED_SOURCES,
             'Missing/extra measured source fingerprints')
    for name, value in sources.items():
        _digest(value, name)
        path = source_root / name
        _require(path.is_file(), f'Measured source file is unavailable: {name}')
        raw = path.read_bytes()
        # A Git checkout can materialize the same source text with LF or CRLF.
        # Keep the original hash; recognize only these equivalent text versions.
        lf = raw.replace(b'\r\n', b'\n')
        hashes = {hashlib.sha256(data).hexdigest() for data in (raw, lf, lf.replace(b'\n', b'\r\n'))}
        _require(value in hashes, f'Recorded source fingerprint differs from source checkout: {name}')


def build_summary(protocol_path: Path, paths: list[Path], root: Path = ROOT,
                  source_root: Path = ROOT) -> dict:
    root = root.resolve()
    source_root = source_root.resolve()
    protocol_path = protocol_path.resolve()
    paths = [path.resolve() for path in paths]
    declaration, declaration_hash = _read(protocol_path)
    _require(declaration['schema'] == 1 and declaration['tolerances'] == TOLERANCES, 'Invalid declaration or changed tolerances')
    _require(declaration['device'] == 'cuda' and declaration['backend'] == 'reference', 'Unexpected diagnostic policy')
    _require(declaration['treatments'] == list(POLICIES), 'Unexpected treatment set/order')
    expected = {row['ratio']: row for row in declaration['checkpoints']}
    _require(len(expected) == 3 and set(expected) == {'1:3', '1:7', '1:15'}, 'Expected three registered hybrid checkpoints')
    _require(len(paths) == len(expected), 'Missing or extra trained-checkpoint reports')
    rows, registry, probes, reports = [], [], [], []
    seen = set()
    for path in paths:
        report, digest = _read(path)
        protocol = report['protocol']
        ratio = protocol['model_config']['ratio']
        _require(ratio in expected and ratio not in seen, 'Duplicate or unexpected checkpoint arm')
        seen.add(ratio)
        _require(report['schema'] == 1 and report['execution_status'] == 'completed' and report['certified'] is False, 'Incomplete or certified diagnostic')
        _require(report['protocol_sha256'] == hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(), 'Altered protocol hash')
        identity = protocol['checkpoint']
        _require(identity['sha256'] == expected[ratio]['sha256'] and identity['path'] == expected[ratio]['path']
                 and identity['path_scope'] == 'repository-relative', 'Checkpoint differs from declaration')
        for name in ('seed', 'device', 'backend', 'chunk_size', 'batch_size', 'tolerances'):
            _require(protocol[name] == declaration[name], f'Changed protocol setting: {name}')
        _require(protocol['treatments'] == list(POLICIES.values()), 'Changed precision treatment')
        for name, value in declaration['precision_flags'].items():
            _require(protocol['runtime']['precision_flags'][name] == value, f'Changed precision flag: {name}')
        config = protocol['model_config']
        _require((config['d_model'], config['n_layers'], config['vocab_size'], config['d_state']) == (448, 16, 16000, 128), 'Unexpected checkpoint geometry')
        inputs = {row['length']: row for row in protocol['inputs']}
        _require(len(inputs) == len(protocol['inputs']) and list(inputs) == declaration['lengths'], 'Changed/duplicate input lengths')
        _digest(protocol['weights_sha256'], 'weights')
        for row in inputs.values():
            for field in ('tokens_sha256', 'targets_sha256'):
                _digest(row[field], field)
        combinations = {(case['length'], case['treatment']) for case in report['cases']}
        planned = {(length, treatment) for length in declaration['lengths'] for treatment in declaration['treatments']}
        _require(combinations == planned and len(report['cases']) == len(planned), 'Missing/duplicate model case')
        for case in report['cases']:
            _validate_case(case, protocol, inputs, case['treatment'])
        _require([probe['length'] for probe in report['exploratory_probes']] == declaration['lengths'], 'Missing/duplicate scan probe')
        for probe in report['exploratory_probes']:
            _validate_probe(probe, probe['length'])
        passed = all(case['passed'] for case in report['cases']) and all(case['passed'] for probe in report['exploratory_probes']
            for group in [probe['scan'], probe['scan']['direct_gradients'], *probe['projections'].values()] for case in group['cases'])
        _require(report['status'] == ('completed' if passed else 'completed_with_parity_failures'), 'Misleading execution status')
        for name in ('wall_seconds', 'peak_allocated_mib'):
            value = report['diagnostic_resources'][name]
            _require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'Invalid actual resource measurement')
        validate_sources(protocol['source_sha256'], source_root)
        registry.append({'ratio': ratio, 'path': path.relative_to(root).as_posix(), 'sha256': digest,
                         'canonical_sha256': _canonical_hash(report), 'protocol_sha256': report['protocol_sha256'],
                         'checkpoint_sha256': identity['sha256'], 'diagnostic_resources': report['diagnostic_resources']})
        for treatment in declaration['treatments']:
            cases = [case for case in report['cases'] if case['treatment'] == treatment]
            rows.append({'ratio': ratio, 'treatment': treatment, 'cases': len(cases),
                'internal': _count([case['comparisons']['internal_passed'] for case in cases]),
                'fp32_anchor': None if treatment == 'fp32_reference' else _count([case['comparisons']['fp32_anchor']['passed'] for case in cases]),
                'diagnostic_all': _count([case['passed'] for case in cases]),
                'failed_lengths': [case['length'] for case in cases if not case['passed']],
                'first_failed_stages': [{'length': case['length'], 'stage': case['first_failed_layer_stage']} for case in cases if case['first_failed_layer_stage'] is not None]})
        for autocast in (True, False):
            groups = {}
            for name in ('scan', 'direct_gradients', 'mixer_projection', 'lm_projection'):
                values = []
                for probe in report['exploratory_probes']:
                    group = {'scan': probe['scan'], 'direct_gradients': probe['scan']['direct_gradients'],
                             'mixer_projection': probe['projections']['mixer_output'], 'lm_projection': probe['projections']['lm_head']}[name]
                    values.append(next(case for case in group['cases'] if case['autocast'] == autocast))
                groups[name] = _group_summary(values)
            probes.append({'ratio': ratio, 'autocast': autocast, **groups})
        reports.append(report)
    _require(seen == set(expected), 'Missing checkpoint arm')
    for report in reports[1:]:
        for key in ('source_sha256', 'runtime'):
            _require(report['protocol'][key] == reports[0]['protocol'][key], f'Inconsistent {key}')
        _require(report['protocol']['inputs'] == reports[0]['protocol']['inputs'], 'Inputs are not paired across checkpoints')
    return {'schema': 1, 'kind': 'trained_checkpoint_precision', 'date': declaration['date'], 'certified': False,
        'status': 'completed' if all(report['status'] == 'completed' for report in reports) else 'completed_with_parity_failures',
        'declaration': {'path': protocol_path.relative_to(root).as_posix(), 'sha256': declaration_hash},
        'rows': sorted(rows, key=lambda row: (list(expected).index(row['ratio']), declaration['treatments'].index(row['treatment']))),
        'raw_reports': registry, 'exploratory_probes': probes, 'tolerances': TOLERANCES, 'runtime': reports[0]['protocol']['runtime'],
        'source_sha256': reports[0]['protocol']['source_sha256'],
        'limits': declaration['limits'] + ['Saved token and target hashes verify pairing within treatments and across checkpoints; different trained weights and architectures still prevent isolating attention alone.',
            'Resource measurements cover the complete diagnostic workload and cannot rank model throughput.', 'Passing four FP32 cases does not certify all shapes, full-model gradients or language quality.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    _require(args.output.resolve() not in {args.protocol.resolve(), *(path.resolve() for path in args.reports)}, 'Cannot overwrite source evidence')
    _require(not args.output.exists(), 'Summary output already exists')
    result = build_summary(args.protocol, args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'rows': len(result['rows']), 'certified': False}))


if __name__ == '__main__':
    main()
