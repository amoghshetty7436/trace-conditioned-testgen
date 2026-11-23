"""Check frozen experiment inputs, pilot exclusion and CPU budget before work."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
manifest = json.loads((root/'manifest.json').read_text())
assert subprocess.check_output(['git', '-C', str(root/'repo'), 'rev-parse', 'HEAD'], text=True).strip() == manifest['commit']
assert not subprocess.check_output(['git', '-C', str(root/'repo'), 'status', '--short'], text=True).strip()
for name, expected in manifest['file_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
remaining = (root/'remaining-tasks.txt').read_text().strip().split(',')
excluded = set(json.loads((root/'excluded-pilot-tasks.json').read_text()))
assert len(remaining) == len(set(remaining)) == 149
assert len(excluded) == 20 and not set(remaining) & excluded
pilot = Path('/pscratch/sd/i/ics35/ttg-kimi-pilot-amogh-20261115')
assert (pilot/'launcher-exit.txt').read_text().strip() == '0'
records = [json.loads(x) for x in (pilot/'pilot-data/tester_runs_kimi.jsonl').read_text().splitlines()]
assert len({(r['trajectory_id'], r['condition'], r['seed']) for r in records}) == 240
budget = json.loads((root/'budget-snapshot.json').read_text())
# The login-node snapshot conservatively adds the entire pilot elapsed time to
# Iris's displayed charge, even if Iris has already accounted for some of it.
assert budget['conservative_remaining_hours'] >= 6.6, 'Insufficient CPU balance for the capped full run'
print('Verified source, full-pool inputs, 149 new tasks, 20 excluded pilot tasks, completed pilot and CPU balance.')
