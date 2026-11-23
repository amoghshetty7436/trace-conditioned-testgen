"""Small Azure capability probe. Credentials are read privately, never logged."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = args.key.read_text().strip()
    endpoint = 'https://ishaan-oai.services.ai.azure.com/openai/v1/chat/completions'
    messages = [{'role': 'user', 'content': 'What is 17 times 19? Reply only with the number.'}]
    for step in range(2):
        body = {'model': 'Kimi-K3.1-Pro', 'messages': messages, 'reasoning_effort': 'high',
                'max_tokens': 65536, 'temperature': 0.2, 'seed': 0,
                'stream': True, 'stream_options': {'include_usage': True}}
        started = time.time()
        record = {'request': body, 'started_at': started}
        request = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                         headers={'api-key': key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                record['http_status'] = response.status
                record['headers'] = {k: v for k, v in response.headers.items()
                                     if k.lower().startswith(('x-ratelimit', 'x-request-id', 'apim-request-id'))}
                chunks = []
                with (args.out / f'probe-{step+1}.sse').open('wb') as raw:
                    for line in response:
                        raw.write(line)
                        line = line.decode().strip()
                        if line.startswith('data:') and line != 'data: [DONE]':
                            chunks.append(json.loads(line[5:]))
                content, reasoning, finish = [], [], []
                for chunk in chunks:
                    if chunk.get('usage'):
                        record['usage'] = chunk['usage']
                    if chunk.get('model'):
                        record['returned_model'] = chunk['model']
                    for choice in chunk.get('choices') or []:
                        delta = choice.get('delta') or {}
                        content.append(delta.get('content') or '')
                        reasoning.append(delta.get('reasoning_content') or '')
                        if choice.get('finish_reason'):
                            finish.append(choice['finish_reason'])
                record.update(content=''.join(content), reasoning_content=''.join(reasoning), finish_reasons=finish)
        except urllib.error.HTTPError as error:
            record.update(http_status=error.code, error=error.read().decode().replace(key, '[REDACTED]'))
        record['elapsed_seconds'] = time.time() - started
        (args.out / f'probe-{step+1}.json').write_text(json.dumps(record, indent=2))
        print(json.dumps({k: v for k, v in record.items() if k not in ('request', 'reasoning_content', 'headers')} |
                         {'reasoning_chars': len(record.get('reasoning_content', ''))}), flush=True)
        if record['http_status'] != 200 or not record.get('content') or not record.get('reasoning_content'):
            raise SystemExit('Probe did not verify both final answer and reasoning text; pilot not submitted.')
        # Match the unchanged harness: only final assistant content is replayed.
        messages.extend([{'role': 'assistant', 'content': record['content']},
                         {'role': 'user', 'content': 'Add 1 to that result. Reply only with the number.'}])


if __name__ == '__main__':
    main()
