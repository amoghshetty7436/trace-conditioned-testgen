import os
import json
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import llm
import tester


class ExperimentTests(unittest.TestCase):
    @patch.dict(os.environ, {"AZURE_OPENAI_BASE_URL": "https://example.invalid/openai/v1",
                             "AZURE_OPENAI_API_KEY": "test-secret",
                             "AZURE_OPENAI_DEPLOYMENT": "gpt-5.6-sol",
                             "AZURE_OPENAI_REASONING_EFFORT": "high"}, clear=True)
    def test_azure_request_and_reasoning_payload(self):
        req = llm.make_request("/chat/completions", {})
        self.assertEqual(req.get_header("Api-key"), "test-secret")
        self.assertTrue(req.full_url.endswith("?api-version=preview"))
        with patch.object(llm, "stream", return_value={"text": "OK", "usage": {}}) as stream:
            llm.chat([{"role": "user", "content": "hello"}], seed=0)
        body = stream.call_args.args[0]
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertNotIn("temperature", body)
        self.assertNotIn("max_tokens", body)
        self.assertIn("max_completion_tokens", body)

    @patch.dict(os.environ, {"TESTER_BASE_URL": "https://example.invalid/v1",
                             "TESTER_API_KEY": "other-secret", "TESTER_MODEL": "other-model",
                             "AZURE_OPENAI_BASE_URL": "https://azure.invalid/openai/v1",
                             "AZURE_OPENAI_REASONING_EFFORT": "high"}, clear=True)
    def test_explicit_tester_endpoint_does_not_inherit_azure_auth(self):
        req = llm.make_request("/chat/completions", {})
        self.assertEqual(req.get_header("Authorization"), "Bearer other-secret")
        self.assertIsNone(llm.settings()["reasoning_effort"])

    def test_infrastructure_failure_cannot_count_as_bug(self):
        good = {"results": {"test_ttg_generated.py::test_bug": "PASSED"}, "output": "ok", "patch_failed": False}
        bad = {"results": {}, "output": "RUN_ERROR timeout", "patch_failed": False}
        with patch.object(tester, "run_file", side_effect=[good, bad, good]):
            result = tester.grade("image", SimpleNamespace(patch="gold"), SimpleNamespace(model_patch="agent"), "code", "path", "pytest")
        self.assertIn("Pa", result["grade_error"])
        self.assertNotIn("flags", result)

    def test_real_failure_and_golden_pass_count_as_bug(self):
        def result(status):
            return {"results": {"test_ttg_generated.py::test_bug": status}, "output": "ok", "patch_failed": False}
        with patch.object(tester, "run_file", side_effect=[result("FAILED"), result("FAILED"), result("PASSED")]):
            graded = tester.grade("image", SimpleNamespace(patch="gold"), SimpleNamespace(model_patch="agent"), "code", "path", "pytest")
        self.assertEqual(graded["flags"], 1)

    def test_code_upgrade_does_not_allow_a_different_model(self):
        old = {"model": "one", "source_sha256": {"tester.py": "old"}}
        new = {"model": "one", "source_sha256": {"tester.py": "new"}}
        with self.assertRaises(SystemExit):
            tester.check_manifest(old, new)
        tester.check_manifest(old, new, allow_code_change=True)
        with self.assertRaises(SystemExit):
            tester.check_manifest(old, {**new, "model": "two"}, allow_code_change=True)

    def test_reply_cache_requires_identical_request(self):
        with tempfile.TemporaryDirectory() as work, patch.object(llm, "chat", return_value={"text": "reply"}) as chat:
            path = Path(work) / "reply.json"
            a = [{"role": "user", "content": "A"}]
            b = [{"role": "user", "content": "B"}]
            self.assertFalse(tester.cached_chat(path, a, 0)["local_cache_hit"])
            self.assertTrue(tester.cached_chat(path, a, 0)["local_cache_hit"])
            self.assertEqual(chat.call_count, 1)
            self.assertFalse(tester.cached_chat(path, b, 0)["local_cache_hit"])
            self.assertEqual(chat.call_count, 2)

    def test_grade_retry_resumes_without_new_generation_or_feedback_runs(self):
        task = SimpleNamespace(instance_id="task", problem_statement="issue", period="new", patch="gold")
        trace = SimpleNamespace(trajectory_id="trace", model_patch="agent", resolved=0)
        ctx = {"test_dir": "tests", "files": {}, "related": {}, "readme": ""}
        reply = {"text": "```python\ndef test_x():\n    assert True\n```", "prompt_tokens": 10, "completion_tokens": 10}
        with tempfile.TemporaryDirectory() as work, patch.object(llm, "chat", return_value=reply) as chat, \
                patch.object(tester, "run_file", return_value={"coverage": None, "output": "PASSED tests/test_ttg_generated.py::test_x"}) as feedback:
            args = (task, trace, "C1", 0, ctx, "image", "pytest", Path(work), "full", {}, {}, {}, {}, "source")
            with patch.object(tester, "grade", side_effect=RuntimeError("interrupted grading")):
                with self.assertRaises(RuntimeError):
                    tester.run_case(*args)
            with patch.object(tester, "grade", return_value={"tests": 1, "flags": 1}):
                result = tester.run_case(*args)
            self.assertEqual(chat.call_count, 1)
            self.assertEqual(feedback.call_count, 1)
            self.assertTrue(result["patch_flagged"])
            self.assertEqual(result["resumed_rounds"], 1)

    def test_container_concurrency_is_bounded_and_isolated(self):
        active = maximum = 0
        names = []
        lock = threading.Lock()

        def fake_docker(*args, **kwargs):
            nonlocal active, maximum
            with lock:
                names.append(args[args.index("--name") + 1])
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.025)
            with lock:
                active -= 1
            return SimpleNamespace(returncode=0, stdout="ok")

        with patch.object(tester, "DOCKER_SLOTS", threading.BoundedSemaphore(2)), patch.object(tester, "docker", side_effect=fake_docker):
            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(lambda _: tester.container("image", "script", {}), range(6)))
        self.assertEqual(results, ["ok"] * 6)
        self.assertEqual(maximum, 2)
        self.assertEqual(len(set(names)), 6)

    def test_concurrent_summary_requests_share_one_generation(self):
        def fake_chat(*args, **kwargs):
            time.sleep(0.025)
            return {"text": "summary"}

        with tempfile.TemporaryDirectory() as work, patch.object(llm, "chat", side_effect=fake_chat) as chat:
            path = Path(work) / "summaries.jsonl"
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: tester.summary_for("trace", [], path), range(4)))
            self.assertEqual(results, ["summary"] * 4)
            self.assertEqual(chat.call_count, 1)
            self.assertEqual(len(path.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
