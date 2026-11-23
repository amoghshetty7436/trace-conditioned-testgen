"""Audit saved pilot artifacts without changing Amogh's experiment or scores."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


def rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines() if line]


def case(row):
    return row['trajectory_id'], row['condition'], row['seed']


def diagnose(records):
    ambiguous = []
    no_code = []
    all_broken = []
    for row in records:
        identity = {k: row[k] for k in ('instance_id', 'trajectory_id', 'condition', 'seed', 'resolved')}
        pa = row.get('statuses', {}).get('Pa', {})
        pg = row.get('statuses', {}).get('Pg', {})
        if row.get('patch_flagged') and not any(
            pa.get(t) in ('FAILED', 'ERROR', 'XPASS') and pg.get(t) in ('PASSED', 'XFAIL')
            for t in row.get('flagged_tests', [])
        ):
            ambiguous.append({**identity, 'flagged_tests': row.get('flagged_tests'),
                              'Pa': pa, 'Pg': pg})
        if not row.get('test_code'):
            no_code.append({**identity, 'completion_tokens': row.get('completion_tokens')})
        if row.get('tests', 0) > 0 and row.get('broken') == row['tests']:
            all_broken.append(identity)
    return {'flagged_without_explicit_failure': ambiguous, 'no_test_code': no_code,
            'all_tests_broken': all_broken,
            'explicit_error_count': sum(bool(r.get('error')) for r in records)}


def main():
    new = rows('pilot-data/tester_runs_deepseek.jsonl')
    old = rows('terra_pilot_baseline.jsonl')
    assert len(new) == len(old) == 240
    assert len({case(r) for r in new}) == 240
    assert {case(r) for r in new} == {case(r) for r in old}
    assert set(Counter(r['instance_id'] for r in new).values()) == {12}
    assert len({r['instance_id'] for r in new}) == 20
    assert len({r['trajectory_id'] for r in new}) == 40
    assert {r['model'] for r in new} == {'DeepSeek-V4-Flash'}
    assert {r['seed'] for r in new} == {0}
    assert {r['condition'] for r in new} == {'C1', 'C2', 'C3', 'C4', 'C5', 'X'}
    checks = rows('pilot-data/task_checks_deepseek.jsonl')
    assert len(checks) == 20 and all(r.get('ok') for r in checks)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    digest = lambda b: hashlib.sha256(b).hexdigest()
    for name, expected in manifest['inputs'].items():
        assert digest((ROOT / 'pilot-data' / name).read_bytes()) == expected
    assert digest((ROOT / 'terra_pilot_baseline.jsonl').read_bytes()) == manifest['baseline_sha256']
    with tarfile.open(ROOT / 'amogh-source.tar.gz') as archive:
        for name, expected in manifest['source_sha256'].items():
            assert digest(archive.extractfile(name).read()) == expected
    assert not (ROOT / 'git-status-after.txt').read_text().strip()
    result = {'integrity_checks': 'passed', 'cases': 240, 'tasks': 20,
              'validation_checks_passed': 20, 'deepseek': diagnose(new), 'terra': diagnose(old),
              'scoring': 'Diagnostics only; original patch_flagged scores preserved.'}
    (ROOT / 'diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v if k not in ('deepseek', 'terra') else {
        name: len(value) if isinstance(value, list) else value for name, value in v.items()
    } for k, v in result.items()}, indent=2))


if __name__ == '__main__':
    main()
