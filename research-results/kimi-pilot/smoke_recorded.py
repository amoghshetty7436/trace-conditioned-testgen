"""Exercise Amogh's exact client through the recorder before allocating a node."""
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
os.umask(0o077)
os.environ['TTG_RECORDER_TOKEN'] = secrets.token_hex(32)
ready = root/'smoke-ready.json'
ready.unlink(missing_ok=True)
with (root/'logs/recorder-smoke.log').open('w') as log:
    process = subprocess.Popen([sys.executable, str(root/'recording_proxy.py'),
                                '--key', '/pscratch/sd/i/ics35/ttg-private/deepseek.key',
                                '--out', str(root/'smoke-api-traces'), '--ready', str(ready)],
                               stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            if ready.exists():
                break
            if process.poll() is not None:
                raise RuntimeError('Recorder exited before becoming ready')
            time.sleep(0.1)
        port = json.loads(ready.read_text())['port']
        os.environ.update(TESTER_BASE_URL=f'http://127.0.0.1:{port}/shard/0/openai/v1',
                          TESTER_API_KEY=os.environ['TTG_RECORDER_TOKEN'],
                          TESTER_MODEL='Kimi-K3.1-Pro', TESTER_REASONING_EFFORT='high',
                          TESTER_MAX_TOKENS='65536')
        sys.path.insert(0, str(root/'repo/src'))
        import llm
        result = llm.chat([{'role': 'user', 'content': 'What is 17 times 19? Reply only with the number.'}], seed=0)
        assert result['text'].strip() == '323', result['text']
        traces = list((root/'smoke-api-traces').glob('*/parsed.json'))
        for _ in range(100):
            if traces:
                break
            time.sleep(0.1)
            traces = list((root/'smoke-api-traces').glob('*/parsed.json'))
        assert len(traces) == 1
        parsed = json.loads(traces[0].read_text())
        assert parsed['content'].strip() == result['text'].strip()
        assert parsed['reasoning_content'] and parsed['sse_done']
        result.update(reasoning_chars=len(parsed['reasoning_content']),
                      response_models=parsed['models'], recorder_verified=True)
        (root/'pilot_api_smoke.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    finally:
        process.terminate()
        process.wait(timeout=10)
        ready.unlink(missing_ok=True)
