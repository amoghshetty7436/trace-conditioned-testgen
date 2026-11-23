"""Verify retry scope and frozen inputs before spending allocation."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
manifest = json.loads((root/'manifest.json').read_text())
assert subprocess.check_output(['git', '-C', str(root/'repo'), 'rev-parse', 'HEAD'], text=True).strip() == manifest['commit']
assert not subprocess.check_output(['git', '-C', str(root/'repo'), 'status', '--short'], text=True).strip()
for name, digest in manifest['file_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
remaining = (root/'remaining-tasks.txt').read_text().strip().split(',')
assert len(remaining) == len(set(remaining)) == 21
assert set(remaining) == set(json.loads((root/'retry-plan.json').read_text())['task_ids'])
assert manifest['conservative_remaining_hours'] >= 2.1
print('Verified unchanged source, full trace pool, 21 skipped tasks and retry budget.')
