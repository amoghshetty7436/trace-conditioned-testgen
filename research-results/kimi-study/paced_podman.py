#!/pscratch/sd/i/ics35/ttg-deepseek-20261031/env/bin/python
"""Pace image pulls only; pass all Podman arguments and I/O through unchanged."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time


def main():
    if sys.argv[1:2] == ['pull']:
        root = Path(__file__).resolve().parent
        with (root/'pull-pacing.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            timestamp = root/'last-pull.txt'
            previous = float(timestamp.read_text()) if timestamp.exists() else 0
            delay = max(0, previous + 60 - time.time())
            if delay:
                time.sleep(delay)
            now = time.time()
            timestamp.write_text(str(now))
            with (root/'pull-events.jsonl').open('a') as events:
                events.write(json.dumps({'started_at': now, 'wait_seconds': delay,
                                         'arguments': sys.argv[1:]})+'\n')
    os.execvp('podman-hpc', ['podman-hpc', *sys.argv[1:]])


if __name__ == '__main__':
    main()
