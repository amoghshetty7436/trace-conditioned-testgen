"""Verify the recorder does not alter requests/responses or leak auth headers."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from recording_proxy import Recorder, summarize

SSE = (b'data: {"model":"Kimi-K3.1","choices":[{"delta":{"reasoning_content":"Reasoning text"}}]}\n\n'
       b'data: {"choices":[{"delta":{"content":"323"},"finish_reason":"stop"}],"usage":{"completion_tokens":8}}\n\n'
       b'data: [DONE]\n\n')


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.server.received = self.rfile.read(int(self.headers['Content-Length']))
        self.server.auth = self.headers.get('api-key')
        self.send_response(self.server.reply_status)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Retry-After', '2')
        self.end_headers()
        self.wfile.write(SSE)


class RecorderTests(unittest.TestCase):
    def test_roundtrip_and_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
            upstream.reply_status = 200
            proxy = ThreadingHTTPServer(('127.0.0.1', 0), Recorder)
            proxy.config = {'token': 'test-token', 'key': 'test-key', 'out': Path(temp),
                            'endpoint': f'http://127.0.0.1:{upstream.server_port}/completion',
                            'required': {'model': 'Kimi-K3.1-Pro', 'reasoning_effort': 'high',
                                         'max_tokens': 65536, 'stream': True}}
            for server in (upstream, proxy):
                threading.Thread(target=server.serve_forever, daemon=True).start()
            url = f'http://127.0.0.1:{proxy.server_port}/shard/3/openai/v1/chat/completions'
            body = json.dumps(proxy.config['required'] | {'messages': [{'role': 'user', 'content': 'hello'}]}).encode()
            request = urllib.request.Request(url, data=body, headers={'Authorization': 'Bearer test-token'})
            try:
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.read(), SSE)
                self.assertEqual(upstream.received, body)
                self.assertEqual(upstream.auth, 'test-key')
                upstream.reply_status = 429
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request)
                self.assertEqual(caught.exception.code, 429)
                self.assertEqual(caught.exception.headers['Retry-After'], '2')
                self.assertEqual(caught.exception.read(), SSE)
                bad = urllib.request.Request(url, data=b'{}', headers={'Authorization': 'Bearer test-token'})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(bad)
                self.assertEqual(caught.exception.code, 422)
                unauth = urllib.request.Request(url, data=body)
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(unauth)
                self.assertEqual(caught.exception.code, 403)
            finally:
                for server in (proxy, upstream):
                    server.shutdown()
                    server.server_close()
            attempts = list(Path(temp).iterdir())
            self.assertEqual(len(attempts), 2)
            for attempt in attempts:
                self.assertEqual((attempt/'request.json').read_bytes(), body)
                self.assertEqual((attempt/'response.sse').read_bytes(), SSE)
                metadata = json.loads((attempt/'metadata.json').read_text())
                self.assertEqual(metadata['shard'], 3)
                self.assertEqual(metadata['request_sha256'], hashlib.sha256(body).hexdigest())
                for file in attempt.iterdir():
                    self.assertNotIn(b'test-key', file.read_bytes())
                    self.assertNotIn(b'test-token', file.read_bytes())
            parsed = summarize(attempts[0]/'response.sse')
            self.assertEqual(parsed['reasoning_content'], 'Reasoning text')
            self.assertEqual(parsed['content'], '323')
            self.assertTrue(parsed['sse_done'])


if __name__ == '__main__':
    unittest.main()
