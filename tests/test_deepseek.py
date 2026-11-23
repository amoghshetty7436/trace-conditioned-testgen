import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "perlmutter"))
import llm
from report_deepseek import outcome, stats


class DeepSeekTests(unittest.TestCase):
    @patch.dict(os.environ, {"AZURE_OPENAI_BASE_URL": "https://example.services.ai.azure.com/openai/v1",
                            "AZURE_OPENAI_API_VERSION": "", "TESTER_API_KEY": "test-secret",
                            "TESTER_PROVIDER": "deepseek", "TESTER_MODEL": "DeepSeek-V4-Flash",
                            "TESTER_THINKING": "enabled", "TESTER_REASONING_EFFORT": "high",
                            "TESTER_MAX_TOKENS": "65536"}, clear=True)
    def test_azure_deepseek_payload_and_usage(self):
        req = llm.make_request("/chat/completions", {})
        self.assertEqual(req.get_header("Api-key"), "test-secret")
        self.assertNotIn("api-version", req.full_url)
        result = {"text": "OK", "reasoning_content": "reason", "finish_reason": "stop",
                  "usage": {"prompt_cache_hit_tokens": 12, "prompt_tokens": 20, "completion_tokens": 10}}
        with patch.object(llm, "stream", return_value=result) as stream:
            reply = llm.chat([{"role": "user", "content": "hello"}], seed=0)
        body = stream.call_args.args[0]
        self.assertNotIn("thinking", body)  # Azure rejects the native DeepSeek field.
        self.assertEqual(body["max_tokens"], 65536)
        self.assertEqual(body["reasoning_effort"], "high")
        for field in ("seed", "temperature", "max_completion_tokens"):
            self.assertNotIn(field, body)
        self.assertEqual(reply["cached_prompt_tokens"], 12)
        self.assertEqual(reply["reasoning_content"], "reason")

    def test_stream_keeps_reasoning_separate_and_rejects_truncation(self):
        def sse(value):
            return ("data: " + json.dumps(value) + "\n").encode()
        data = sse({"choices": [{"delta": {"reasoning_content": "reason"}}]})
        data += sse({"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 2}})
        with patch.object(llm.urllib.request, "urlopen", return_value=io.BytesIO(data)):
            result = llm.stream({"messages": []})
        self.assertEqual((result["text"], result["reasoning_content"]), ("answer", "reason"))
        with patch.object(llm.urllib.request, "urlopen", return_value=io.BytesIO(sse({"choices": [{"delta": {"content": "partial"}}]}))):
            with self.assertRaises(OSError):
                llm.stream({"messages": []})

    def test_comparison_does_not_score_skips_or_missing_tests_as_catches(self):
        row = {"test_code": "code", "statuses": {"P0": {"t": "FAILED"}, "Pa": {"t": "SKIPPED"}, "Pg": {"t": "PASSED"}}}
        self.assertIsNone(outcome(row)[1])
        row["statuses"]["Pa"] = {"other": "PASSED"}
        self.assertIsNone(outcome(row)[1])
        row["statuses"]["Pa"] = {"t": "FAILED"}
        self.assertEqual(outcome(row), ("scored", True))
        row.update(condition="C1", resolved=0)
        result = stats([row, {"condition": "C1", "resolved": 0, "test_code": None}])
        self.assertEqual(result["C1"]["bug_caught"]["rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
