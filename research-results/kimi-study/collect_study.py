"""Back up the queued study; never starts or resubmits a job."""
import json
from pathlib import Path
import subprocess
import time

dest = Path(__file__).resolve().parent
host = 'ics35@perlmutter.nersc.gov'
socket = '/Users/ishaan812/.ssh/ttg-perlmutter-ics35.sock'
root = '/pscratch/sd/i/ics35/ttg-kimi-study-amogh-20261115'
job = '61373662'
ssh = ['ssh', '-S', socket, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host]
deadline = time.monotonic() + 3 * 24 * 3600
while time.monotonic() < deadline:
    try:
        state = subprocess.check_output(ssh + [f"squeue -h -j {job} -o '%T'"], text=True, timeout=40).strip()
        subprocess.run(['rsync', '-az', '-e', f'ssh -S {socket} -o BatchMode=yes -o ConnectTimeout=15',
                        '--exclude', 'repo/', '--exclude', 'recorder-ready.json',
                        f'{host}:{root}/', str(dest) + '/'], check=True, timeout=600)
        path = dest/'study-data/tester_runs_kimi_main.jsonl'
        records = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        unique = {(r['trajectory_id'], r['condition'], r['seed']) for r in records}
        status = {'job_id': job, 'state': state or 'ENDED', 'saved_cases': len(unique),
                  'planned_cases': 1788, 'updated_at': time.time()}
        (dest/'live-status.json').write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status), flush=True)
        if not state:
            info = subprocess.check_output(ssh + [f'sacct -j {job} -X -P -o JobID,State,ElapsedRaw,AllocNodes,ExitCode'], text=True, timeout=40)
            (dest/'final-accounting.txt').write_text(info)
            (dest/'collection-complete.json').write_text(json.dumps(status, indent=2)+'\n')
            break
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        print(type(error).__name__+': '+str(error), flush=True)
    time.sleep(120)
else:
    raise SystemExit('Collection deadline reached; no job was resubmitted.')
