"""Back up and report one authorized pilot. Does not submit or resume jobs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

DEST = Path(__file__).resolve().parent
REMOTE = '/pscratch/sd/i/ics35/ttg-kimi-pilot-amogh-20261115'
HOST = 'ics35@perlmutter.nersc.gov'
SOCKET = '/Users/ishaan812/.ssh/ttg-perlmutter-ics35.sock'
JOB = '61370723'
SSH = ['ssh', '-S', SOCKET, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST]


def remote(command):
    return subprocess.run(SSH + [command], capture_output=True, text=True, timeout=90, check=True).stdout


def sync():
    subprocess.run(['rsync', '-az', '-e', f'ssh -S {SOCKET} -o BatchMode=yes -o ConnectTimeout=15',
                    '--exclude', 'repo/', '--exclude', 'recorder-ready.json',
                    f'{HOST}:{REMOTE}/', str(DEST) + '/'], timeout=180, check=True)


def records(path):
    if not path.exists():
        return []
    result = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1:
                raise
    return result


def key(row):
    return row['trajectory_id'], row['condition'], row['seed']


def score(rows):
    result = {}
    for condition in ('C1', 'C2', 'C3', 'C4', 'C5', 'X'):
        result[condition] = {}
        for label, resolved in (('catches', 0), ('false_alarms', 1)):
            group = [r for r in rows if r['condition'] == condition and r['resolved'] == resolved]
            result[condition][label] = {'flags': sum(bool(r.get('patch_flagged')) for r in group), 'cases': len(group)}
    return result


def diagnostics(rows):
    result = {'no_test_code': [], 'all_tests_broken': [], 'ambiguous_flags': []}
    for row in rows:
        identity = {k: row[k] for k in ('instance_id', 'trajectory_id', 'condition', 'seed', 'resolved')}
        if not row.get('test_code'):
            result['no_test_code'].append(identity)
        if row.get('tests', 0) > 0 and row.get('broken') == row['tests']:
            result['all_tests_broken'].append(identity)
        pa = row.get('statuses', {}).get('Pa', {})
        pg = row.get('statuses', {}).get('Pg', {})
        if row.get('patch_flagged') and not any(pa.get(t) in ('FAILED', 'ERROR', 'XPASS')
                                                and pg.get(t) in ('PASSED', 'XFAIL')
                                                for t in row.get('flagged_tests', [])):
            result['ambiguous_flags'].append(identity)
    return result


def report(state):
    raw = records(DEST/'pilot-data/tester_runs_kimi.jsonl')
    current = {key(r): r for r in raw}
    baseline = {label: {key(r): r for r in records(DEST/file)} for label, file in (
        ('terra', 'terra_pilot_baseline.jsonl'), ('deepseek', 'deepseek_pilot_baseline.jsonl'))}
    common = current.keys() & baseline['terra'].keys() & baseline['deepseek'].keys()
    paired = {'kimi': score([current[k] for k in common])}
    paired.update({label: score([data[k] for k in common]) for label, data in baseline.items()})
    checks = {r['instance_id']: r for r in records(DEST/'pilot-data/task_checks_kimi.jsonl')}
    api = Counter()
    for path in (DEST/'api-traces').glob('*/parsed.json'):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        api['parsed_calls'] += 1
        api['calls_with_reasoning'] += bool(record.get('reasoning_content'))
        api['incomplete_streams'] += not record.get('sse_done')
        api['length_finishes'] += 'length' in record.get('finish_reasons', [])
        for field in ('prompt_tokens', 'completion_tokens'):
            api[field] += record.get('usage', {}).get(field) or 0
        usage = record.get('usage', {})
        api['reasoning_tokens'] += usage.get('reasoning_tokens') or (usage.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
    result = {'job_id': JOB, 'state': state, 'planned_cases': 240, 'saved_cases': len(current),
              'duplicate_cases': len(raw)-len(current), 'paired_cases': len(common),
              'complete_tasks': sum(v == 12 for v in Counter(r['instance_id'] for r in current.values()).values()),
              'validation_checks': len(checks), 'excluded_tasks': [k for k, r in checks.items() if not r.get('ok')],
              'paired_scores': paired, 'diagnostics': diagnostics(list(current.values())),
              'api_trace_totals': dict(api), 'full_study_authorized': False}
    (DEST/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    fmt = lambda s: f"{s['flags']}/{s['cases']}"
    lines = ['# Kimi Azure pilot', '', f'Job `{JOB}`: **{state}**; {len(current)}/240 cases saved.', '',
             'Amogh unchanged commit `44ec6b20afeb790a8c5c625497c89bf0f163967d`; 20 tasks, seed 0, six conditions.',
             'Azure deployment `Kimi-K3.1-Pro`; smoke response model `Kimi-K3.1`; reasoning high; output cap 65,536.', '',
             '| Condition | Kimi catches | DeepSeek catches | Terra catches | Kimi false alarms | DeepSeek false alarms | Terra false alarms |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for condition in paired['kimi']:
        values = [fmt(paired[model][condition][label]) for label in ('catches', 'false_alarms') for model in ('kimi', 'deepseek', 'terra')]
        lines.append('| '+condition+' | '+' | '.join(values)+' |')
    lines += ['', 'Only matching saved cases are compared. Primary scores preserve the upstream harness; missing cases are not counted as completed.',
              f"Task validations: {len(checks)}; exclusions: {len(result['excluded_tasks'])}; complete tasks: {result['complete_tasks']}/20.",
              f"Generation failures with no test code: {len(result['diagnostics']['no_test_code'])}; flags without explicit failing-test evidence: {len(result['diagnostics']['ambiguous_flags'])}.", '',
              f"Recorded API calls: {api['parsed_calls']}; with reasoning text: {api['calls_with_reasoning']}; output-limit finishes: {api['length_finishes']}.",
              f"Recorded API usage (includes summary calls): {api['prompt_tokens']:,} input and {api['completion_tokens']:,} completion tokens, including {api['reasoning_tokens']:,} reported reasoning tokens.", '',
              'Raw API requests, SSE response bodies, extracted reasoning/final text and usage are in `api-traces/`. Authentication headers are not recorded.',
              'The external recorder forwards request and response bodies unchanged. The original harness replays final assistant content only; captured reasoning is not injected into subsequent prompts. This preserves the earlier protocol but differs from Kimi\'s recommendation to replay reasoning in multi-turn conversations.',
              'Baseline caveat: the DeepSeek pilot has one skip-related ambiguous catch and two generations without test code. Repeated condition-level observations are not independent tasks. Reasoning labels do not establish equal inference compute.', '',
              '**Pilot only. Do not start the full study without explicit approval after review.**']
    accounting = DEST/'accounting.txt'
    if accounting.exists():
        lines += ['', 'Slurm accounting:', '```', accounting.read_text().strip(), '```']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def final_audit(result):
    manifest = json.loads((DEST/'manifest.json').read_text())
    failures = []
    for name, expected in manifest['inputs'].items():
        if hashlib.sha256((DEST/'pilot-data'/name).read_bytes()).hexdigest() != expected:
            failures.append('Input hash mismatch: '+name)
    if (DEST/'git-status-after.txt').read_text().strip():
        failures.append('Remote source status is not clean')
    if result['saved_cases'] != 240 or result['paired_cases'] != 240 or result['duplicate_cases']:
        failures.append('Expected 240 unique paired cases')
    if result['validation_checks'] != 20 or result['excluded_tasks']:
        failures.append('Not all 20 task validations passed')
    remote_hash = (DEST/'raw-result-sha256.txt').read_text().split()[0]
    if hashlib.sha256((DEST/'pilot-data/tester_runs_kimi.jsonl').read_bytes()).hexdigest() != remote_hash:
        failures.append('Local/remote result hash mismatch')
    (DEST/'integrity-audit.json').write_text(json.dumps({'passed': not failures, 'failures': failures}, indent=2)+'\n')
    with (DEST/'REPORT.md').open('a') as report_file:
        report_file.write('\nFinal integrity audit: '+('passed.' if not failures else '; '.join(failures))+'\n')
    hashes = {str(p.relative_to(DEST)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in DEST.rglob('*') if p.is_file() and p.name != 'artifact-sha256.json' and '__pycache__' not in p.parts}
    (DEST/'artifact-sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')


def main():
    previous = None
    deadline = time.monotonic()+3*3600
    while time.monotonic() < deadline:
        try:
            state = remote(f"squeue -h -j {JOB} -o '%T'").strip()
            sync()
            result = report(state or 'FINISHING')
            status = (state, result['saved_cases'], result['api_trace_totals'].get('parsed_calls', 0))
            if status != previous:
                print(json.dumps({'state': status[0], 'saved_cases': status[1], 'recorded_api_calls': status[2]}), flush=True)
                previous = status
            if not state:
                remote(f"cd {REMOTE}/repo; /pscratch/sd/i/ics35/ttg-deepseek-20261031/env/bin/python src/tester.py summarize --data ../pilot-data --out-tag kimi > ../logs/final-summary.log 2>&1; sacct -j {JOB} -X -P -o JobID,Account,State,ElapsedRaw,AllocNodes,Timelimit,ExitCode > ../accounting.txt; iris user ics35 > ../allocation-after.txt; git status --short > ../git-status-after.txt; sha256sum ../pilot-data/tester_runs_kimi.jsonl > ../raw-result-sha256.txt")
                sync()
                state = (DEST/'accounting.txt').read_text().splitlines()[1].split('|')[2]
                result = report(state)
                (DEST/'collection-complete.json').write_text(json.dumps({'state': state, 'saved_cases': result['saved_cases'], 'collected_at': time.time()}, indent=2)+'\n')
                final_audit(result)
                print(f"Collection complete: {state}, {result['saved_cases']}/240 cases", flush=True)
                return
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            print(type(error).__name__+': '+str(error), flush=True)
        time.sleep(60)
    raise SystemExit('Collector deadline reached; no job was resubmitted.')


if __name__ == '__main__':
    main()
