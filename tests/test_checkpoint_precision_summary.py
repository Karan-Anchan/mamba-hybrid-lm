"""A trained-shape summary must retain failures and reject altered evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.summarize_checkpoint_precision import MEASURED_SOURCES, ROOT, build_summary

RESEARCH = Path(__file__).resolve().parents[1] / 'docs/research'


@pytest.fixture
def matrix(tmp_path):
    protocol = tmp_path / 'protocol.json'
    protocol.write_bytes((RESEARCH / 'checkpoint-precision-protocol-2026-10-04.json').read_bytes())
    paths = []
    for original in sorted((RESEARCH / 'checks/trained-precision-2026-10-04').glob('*.json')):
        path = tmp_path / original.name
        path.write_bytes(original.read_bytes())
        paths.append(path)
    return protocol, paths, tmp_path


def test_completed_diagnostic_preserves_the_failed_trained_endpoint(matrix):
    result = build_summary(*matrix)
    assert result['certified'] is False and result['status'] == 'completed_with_parity_failures'
    assert len(result['raw_reports']) == 3 and len(result['rows']) == 15
    reference = next(row for row in result['rows'] if row['ratio'] == '1:15' and row['treatment'] == 'fp32_reference')
    assert reference['internal'] == {'passed': 3, 'total': 4}
    assert reference['failed_lengths'] == [127]
    assert all(row['internal']['passed'] == row['fp32_anchor']['passed'] == 0 for row in result['rows'] if row['fp32_anchor'] is not None)
    reports = [json.loads(path.read_text(encoding='utf-8')) for path in matrix[1]]
    assert all(report['protocol']['inputs'] == reports[0]['protocol']['inputs'] for report in reports)
    assert any('pairing within treatments and across checkpoints' in limit for limit in result['limits'])
    assert any('different trained weights and architectures' in limit for limit in result['limits'])


def test_missing_or_duplicate_trained_arm_is_rejected(matrix):
    protocol, paths, root = matrix
    with pytest.raises(ValueError, match='Missing or extra'):
        build_summary(protocol, paths[:-1], root)
    with pytest.raises(ValueError, match='Duplicate'):
        build_summary(protocol, [paths[0], paths[0], paths[2]], root)


def test_changed_checkpoint_identity_cannot_be_approved(matrix):
    protocol, paths, root = matrix
    declaration = json.loads(protocol.read_text(encoding='utf-8'))
    declaration['checkpoints'][0]['sha256'] = '0' * 64
    protocol.write_text(json.dumps(declaration), encoding='utf-8')
    with pytest.raises(ValueError, match='Checkpoint differs'):
        build_summary(protocol, paths, root)


def test_misleading_green_status_or_nonfinite_resources_is_rejected(matrix):
    protocol, paths, root = matrix
    original = json.loads(paths[0].read_text(encoding='utf-8'))
    changed = deepcopy(original)
    changed['status'] = 'completed'
    paths[0].write_text(json.dumps(changed), encoding='utf-8')
    with pytest.raises(ValueError, match='Misleading'):
        build_summary(protocol, paths, root)
    changed = deepcopy(original)
    changed['diagnostic_resources']['wall_seconds'] = 1e300
    text = json.dumps(changed).replace('1e+300', '1e400')
    paths[0].write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid actual resource'):
        build_summary(protocol, paths, root)


def resign(report):
    report['protocol_sha256'] = hashlib.sha256(json.dumps(report['protocol'], sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize('mutation', ['empty', 'missing', 'extra', 'stale'])
def test_all_consistently_resigned_reports_still_require_exact_measured_sources(matrix, mutation):
    protocol, paths, root = matrix
    for path in paths:
        report = json.loads(path.read_text(encoding='utf-8'))
        sources = report['protocol']['source_sha256']
        if mutation == 'empty':
            sources.clear()
        elif mutation == 'missing':
            sources.pop('scripts/check_scan_backend.py')
        elif mutation == 'extra':
            sources['scripts/unmeasured.py'] = '0' * 64
        else:
            sources['scripts/study_scan_precision.py'] = '0' * 64
        resign(report)
        path.write_text(json.dumps(report), encoding='utf-8')
    expected = 'Recorded source fingerprint' if mutation == 'stale' else 'measured source fingerprints'
    with pytest.raises(ValueError, match=expected):
        build_summary(protocol, paths, root)


def test_source_fingerprints_allow_only_equivalent_lf_crlf_text(matrix, tmp_path):
    protocol, paths, root = matrix
    source_root = tmp_path / 'source-checkout'
    for name in MEASURED_SOURCES:
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        original = (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
        path.write_bytes(original.replace(b'\n', b'\r\n'))
    assert build_summary(protocol, paths, root, source_root)['status'] == 'completed_with_parity_failures'
    stale = source_root / 'src/model/mamba2.py'
    with stale.open('ab') as handle:
        handle.write(b'\n# different source contents\n')
    with pytest.raises(ValueError, match='Recorded source fingerprint'):
        build_summary(protocol, paths, root, source_root)


def test_cross_checkpoint_input_pairing_cannot_be_invented_after_resigning(matrix):
    protocol, paths, root = matrix
    report = json.loads(paths[0].read_text(encoding='utf-8'))
    identity = report['protocol']['inputs'][0]
    identity['tokens_sha256'] = '0' * 64
    for case in report['cases']:
        if case['length'] == identity['length']:
            case['tokens_sha256'] = identity['tokens_sha256']
    resign(report)
    paths[0].write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='Inputs are not paired across checkpoints'):
        build_summary(protocol, paths, root)
