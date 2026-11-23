"""Loopback-only, byte-preserving Azure recorder; never changes benchmark bodies.

Only the authentication header is replaced. Requests and raw response bodies are
saved separately from Amogh's unchanged source and scores. No headers with secrets
are persisted. One directory per API attempt also preserves failed/partial calls.
"""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid


def summarize(raw_path):
    text, reasoning, finishes, models, usage = [], [], [], set(), {}
    done = False
    malformed = 0
    for line in raw_path.read_bytes().splitlines():
        if not line.startswith(b'data:'):
            continue
        data = line[5:].strip()
        if data == b'[DONE]':
            done = True
            continue
        try:
            chunk = json.loads(data)
        except (ValueError, UnicodeError):
            malformed += 1
            continue
        if chunk.get('model'):
            models.add(chunk['model'])
        if chunk.get('usage'):
            usage = chunk['usage']
        for choice in chunk.get('choices') or []:
            delta = choice.get('delta') or {}
            text.append(delta.get('content') or '')
            reasoning.append(delta.get('reasoning_content') or '')
            if choice.get('finish_reason'):
                finishes.append(choice['finish_reason'])
    return {'content': ''.join(text), 'reasoning_content': ''.join(reasoning),
            'finish_reasons': finishes, 'models': sorted(models), 'usage': usage,
            'sse_done': done, 'malformed_chunks': malformed}


class Recorder(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        config = self.server.config
        supplied = self.headers.get('Authorization', '').removeprefix('Bearer ')
        if not hmac.compare_digest(supplied, config['token']):
            self.send_error(403)
            return
        route = re.fullmatch(r'/shard/([0-8])/openai/v1/chat/completions', self.path)
        if not route:
            self.send_error(404)
            return
        n = int(self.headers.get('Content-Length', '0'))
        if not 0 < n <= 32 * 1024 * 1024:
            self.send_error(413)
            return
        body = self.rfile.read(n)
        try:
            payload = json.loads(body)
        except ValueError:
            self.send_error(400)
            return
        # Prevent the upstream client's parameter fallback from silently dropping
        # the registered reasoning setting, model, or output cap.
        if any(payload.get(k) != v for k, v in config['required'].items()):
            self.send_error(422, 'Request does not match registered experiment configuration')
            return
        attempt = config['out'] / (str(time.time_ns()) + '-' + uuid.uuid4().hex[:8])
        attempt.mkdir(mode=0o700)
        (attempt / 'request.json').write_bytes(body)
        metadata = {'started_at': time.time(), 'shard': int(route[1]),
                    'request_sha256': hashlib.sha256(body).hexdigest(),
                    'request_body_forwarded_unchanged': True}
        request = urllib.request.Request(config['endpoint'], data=body,
                                         headers={'api-key': config['key'], 'Content-Type': 'application/json'})
        sent_headers = False
        try:
            try:
                upstream = urllib.request.urlopen(request, timeout=900)
            except urllib.error.HTTPError as error:
                upstream = error
            with upstream:
                metadata['http_status'] = upstream.status
                metadata['headers'] = {k: v for k, v in upstream.headers.items()
                                       if k.lower().startswith(('x-request-id', 'apim-request-id', 'x-ratelimit'))}
                self.send_response(upstream.status)
                for name in ('Content-Type', 'Retry-After'):
                    if upstream.headers.get(name):
                        self.send_header(name, upstream.headers[name])
                self.send_header('Connection', 'close')
                self.end_headers()
                sent_headers = True
                self.close_connection = True
                # Write each original line to disk before forwarding it verbatim.
                with (attempt / 'response.sse').open('wb') as raw:
                    for line in upstream:
                        raw.write(line)
                        raw.flush()
                        self.wfile.write(line)
                        self.wfile.flush()
                    os.fsync(raw.fileno())
        except Exception as error:
            metadata['transport_error_type'] = type(error).__name__
            if not sent_headers:
                self.send_error(502, 'Upstream connection failed')
        finally:
            metadata['ended_at'] = time.time()
            metadata['elapsed_seconds'] = metadata['ended_at'] - metadata['started_at']
            raw_path = attempt / 'response.sse'
            if raw_path.exists():
                metadata['response_sha256'] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                if metadata.get('http_status') == 200:
                    parsed = summarize(raw_path)
                    (attempt / 'parsed.json').write_text(json.dumps(parsed, indent=2))
                    metadata['reasoning_chars'] = len(parsed['reasoning_content'])
                    metadata['content_chars'] = len(parsed['content'])
                    metadata['sse_done'] = parsed['sse_done']
            (attempt / 'metadata.json').write_text(json.dumps(metadata, indent=2))
            print(json.dumps({'attempt': attempt.name, **metadata}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ready', type=Path, required=True)
    ap.add_argument('--port', type=int, default=0)
    args = ap.parse_args()
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Recorder)
    server.config = {'key': args.key.read_text().strip(), 'token': os.environ['TTG_RECORDER_TOKEN'],
                     'out': args.out,
                     'endpoint': 'https://ishaan-oai.services.ai.azure.com/openai/v1/chat/completions',
                     'required': {'model': 'Kimi-K3.1-Pro', 'reasoning_effort': 'high',
                                  'max_tokens': 65536, 'stream': True}}
    args.ready.write_text(json.dumps({'port': server.server_port}))
    server.serve_forever()


if __name__ == '__main__':
    main()
