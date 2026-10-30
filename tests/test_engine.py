"""Offline tests: the engine is exercised against a fake transport, never the real NIM API.

Run from the plugin root (unittest, not pytest — that is what the host venv has):

    ~/.hermes/hermes-agent/venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import json
import os
import socket
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nim import engine, render, watch  # noqa: E402


class FakeResponse(io.BytesIO):
    """Enough of an http.client response for both SSE iteration and json.load()."""

    def __init__(self, chunks: list[bytes] | bytes):
        payload = chunks if isinstance(chunks, bytes) else b"".join(chunks)
        super().__init__(payload)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def __iter__(self):
        return iter(self.readlines())


def http_error(code: int, body: dict | str) -> urllib.error.HTTPError:
    raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    return urllib.error.HTTPError("https://x/v1/chat/completions", code, "err", {}, io.BytesIO(raw))


SSE_OK = b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'


def opener(outcomes):
    """A transport that replays a fixed script: FakeResponse to return, exception to raise.

    A list handed to `side_effect` would be returned item-by-item *without* calling anything, so the
    script is replayed explicitly here.
    """
    queue = list(outcomes)

    def _open(*_args, **_kwargs):
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):  # a factory, so a body can be streamed more than once
            return item()
        return item

    return _open


class AttemptTests(unittest.TestCase):
    def _attempt(self, outcome):
        with mock.patch.object(engine, "_urlopen", side_effect=opener([outcome])):
            return engine.attempt("https://x/v1", "m", "k", 5)

    def test_streamed_token_is_ok(self):
        result = self._attempt(FakeResponse(SSE_OK))
        self.assertEqual(result["kind"], "ok")
        self.assertIsInstance(result["seconds"], float)

    def test_empty_stream_is_stall_not_ok(self):
        self.assertEqual(self._attempt(FakeResponse(b""))["kind"], "stall")

    def test_410_is_eol_and_parses_the_date(self):
        detail = {"detail": "The model 'z-ai/glm-5.3' has reached its end of life on "
                            "2026-10-28T08:00:00Z and is no longer available."}
        result = self._attempt(http_error(410, detail))
        self.assertEqual(result["kind"], "eol")
        self.assertEqual(result["eol_at"], "2026-10-28T08:00:00Z")

    def test_404_is_not_provisioned(self):
        self.assertEqual(self._attempt(http_error(404, {"status": 404}))["kind"],
                         "not_provisioned")

    def test_429_is_rate_limited(self):
        self.assertEqual(self._attempt(http_error(429, {"status": 429}))["kind"], "rate_limited")

    def test_timeout_is_stall(self):
        self.assertEqual(self._attempt(socket.timeout("read timed out"))["kind"], "stall")

    def test_500_is_reported_verbatim(self):
        self.assertEqual(self._attempt(http_error(500, {"status": 500}))["kind"], "http_500")


class ProbeTests(unittest.TestCase):
    def _probe(self, kinds):
        script = {"ok": lambda: FakeResponse(SSE_OK),
                  "stall": socket.timeout("nope"),
                  "429": http_error(429, {"status": 429}),
                  "410": http_error(410, {"detail": "end of life on 2026-01-01T00:00:00Z"})}
        outcomes = [script[kind] for kind in kinds]
        with mock.patch.object(engine, "_urlopen", side_effect=opener(outcomes)):
            return engine.probe("https://x/v1", "m", "k", 5, attempts=len(kinds), stagger=0)

    def test_all_ok(self):
        self.assertEqual(self._probe(["ok", "ok"])["status"], "ok")

    def test_one_stall_is_flaky_not_ok(self):
        result = self._probe(["ok", "stall"])
        self.assertEqual(result["status"], "flaky")
        self.assertEqual(result["ok_attempts"], 1)

    def test_all_stalls(self):
        self.assertEqual(self._probe(["stall", "stall"])["status"], "stall")

    def test_429_alongside_an_answer_is_throttled_not_rot(self):
        self.assertEqual(self._probe(["ok", "429"])["status"], "throttled")

    def test_only_429(self):
        self.assertEqual(self._probe(["429", "429"])["status"], "rate_limited")

    def test_eol_stops_retrying(self):
        result = self._probe(["410", "ok"])
        self.assertEqual(result["status"], "eol")
        self.assertEqual(result["attempts"], 1)


class VerdictTests(unittest.TestCase):
    def test_severity_buckets(self):
        self.assertEqual(engine.severity({"status": "ok"}), "ok")
        self.assertEqual(engine.severity({"status": "flaky"}), "warn")
        self.assertEqual(engine.severity({"status": "throttled"}), "warn")
        self.assertEqual(engine.severity({"status": "stall"}), "dead")
        self.assertEqual(engine.severity({"status": "not_provisioned"}), "dead")
        self.assertEqual(engine.severity({"status": "http_500"}), "dead")

    def test_labels_are_sentences(self):
        self.assertIn("zero bytes", engine.label(
            {"status": "stall", "attempts": 3, "trials": ["stall:25"] * 3}))
        self.assertIn("EOL", engine.label({"status": "eol", "eol_at": "2026-10-28T08:00:00Z"}))
        self.assertIn("404", engine.label({"status": "not_provisioned", "attempts": 1}))


class RankTests(unittest.TestCase):
    def test_param_count_comes_from_the_id(self):
        self.assertEqual(engine.params_from_id("nvidia/nemotron-3-ultra-550b-a55b"), 550)
        self.assertEqual(engine.params_from_id("nvidia/nemotron-3.5-lightning-30b-a3b"), 30)
        self.assertEqual(engine.params_from_id("openai/gpt-oss-20b"), 20)
        self.assertEqual(engine.params_from_id("google/gemma-4-31b-it"), 31)
        self.assertIsNone(engine.params_from_id("minimaxai/minimax-m3.1-flash"))
        self.assertIsNone(engine.params_from_id("nvidia/nemotron-parse-2.0"))

    def _result(self, model, ok_attempts=2, attempts=2, ttft=0.5, status="ok"):
        return {"model": model, "status": status, "ok_attempts": ok_attempts,
                "attempts": attempts, "ttft_s": ttft, "trials": []}

    def test_bigger_model_ranks_first_despite_slower(self):
        big = self._result("nvidia/nemotron-3-ultra-550b-a55b", ttft=2.0)
        small = self._result("openai/gpt-oss-20b", ttft=0.2)
        ordered = sorted([small, big], key=engine.rank_key)
        self.assertEqual(ordered[0]["model"], big["model"])

    def test_unknown_size_sorts_last_and_is_not_guessed(self):
        # neither a published score nor a size in the id: it must not be handed either
        unknown = self._result("vendor/unnamed-model", ttft=0.1)
        small = self._result("openai/gpt-oss-20b", ttft=5.0)
        ordered = sorted([unknown, small], key=engine.rank_key)
        self.assertEqual(ordered[0]["model"], small["model"])

    def test_reliability_breaks_a_size_tie(self):
        steady = self._result("a/model-30b", ok_attempts=2, attempts=2, ttft=3.0)
        shaky = self._result("b/model-30b", ok_attempts=1, attempts=2, ttft=0.1,
                             status="flaky")
        ordered = sorted([shaky, steady], key=engine.rank_key)
        self.assertEqual(ordered[0]["model"], steady["model"])

    def test_latency_breaks_a_full_tie(self):
        slow = self._result("a/model-30b", ttft=4.0)
        fast = self._result("b/model-30b", ttft=0.4)
        ordered = sorted([slow, fast], key=engine.rank_key)
        self.assertEqual(ordered[0]["model"], fast["model"])

    def test_default_view_is_ordered_best_first_and_names_the_pick(self):
        report = _fake_report({
            "openai/gpt-oss-20b": "ok",
            "nvidia/nemotron-3-ultra-550b-a55b": "ok",
            "minimaxai/minimax-m3.1-flash": "ok",
        })
        for model in report["models"]:
            report["models"][model]["attempts"] = 2
            report["models"][model]["ok_attempts"] = 2
            report["models"][model]["ttft_s"] = 0.5
        text = render.render_cached(report)
        # one line per model, org prefix dropped, the top pick marked with a star
        first_model = next(line for line in text.splitlines() if line.startswith("★"))
        # minimax carries a published AA index (29) and nemotron a lower one (23), so the published
        # score outranks the bigger parameter count — this is the case the size prior got wrong.
        self.assertIn("minimax-m3.1-flash", first_model)
        self.assertNotIn("minimaxai/", first_model)
        self.assertIn("★ recommended: minimaxai/minimax-m3.1-flash", text)
        # model rows stay narrow enough not to wrap twice on a phone (the footer may run longer:
        # it is the one line carrying a full id you would paste into config.yaml)
        rows = [line for line in text.splitlines()
                if line[:2] in {"★ ", "✓ ", "⚠ ", "✗ "}
                and not line.startswith("★ recommended")]
        self.assertTrue(rows)
        self.assertLess(max(len(row) for row in rows), 70)


class BenchmarkPriorTests(unittest.TestCase):
    def test_published_index_beats_a_bigger_parameter_count(self):
        published = {"model": "minimaxai/minimax-m3.1-flash", "status": "ok", "ok_attempts": 2,
                     "attempts": 2, "ttft_s": 4.0, "trials": []}
        bigger = {"model": "nvidia/nemotron-3-ultra-550b-a55b", "status": "ok",
                  "ok_attempts": 2, "attempts": 2, "ttft_s": 0.5, "trials": []}
        ordered = sorted([bigger, published], key=engine.rank_key)
        self.assertEqual(ordered[0]["model"], published["model"])
        # and the bigger model still outranks a smallest one with no published score
        small = {"model": "openai/gpt-oss-20b", "status": "ok", "ok_attempts": 2,
                 "attempts": 2, "ttft_s": 0.2, "trials": []}
        self.assertEqual(sorted([small, bigger], key=engine.rank_key)[0]["model"], bigger["model"])

    def test_inherited_scores_are_labelled_as_inherited(self):
        from nim import benchmarks

        flash = benchmarks.lookup("minimaxai/minimax-m3.1-flash")
        self.assertIsNotNone(flash)
        self.assertTrue(flash.inherited)
        self.assertIn("family", benchmarks.describe("minimaxai/minimax-m3.1-flash"))
        self.assertFalse(benchmarks.lookup("nvidia/nemotron-3-ultra-550b-a55b").inherited)

    def test_unknown_models_get_no_score(self):
        from nim import benchmarks

        self.assertIsNone(benchmarks.lookup("meta/muse-glimmer-30b"))
        self.assertEqual(benchmarks.describe("meta/muse-glimmer-30b"), "")

    def test_recommendation_is_gated_on_speed_not_just_rank(self):
        slow = {"model": "minimaxai/minimax-m3.1-flash", "status": "ok", "ok_attempts": 2,
                "attempts": 2, "ttft_s": 9.0, "trials": []}   # best score, too slow
        quick = {"model": "nvidia/nemotron-3-ultra-550b-a55b", "status": "ok",
                 "ok_attempts": 2, "attempts": 2, "ttft_s": 0.6, "trials": []}
        report = {"models": {slow["model"]: slow, quick["model"]: quick}}
        self.assertEqual(engine.recommended(report)["model"], quick["model"])
        # flip the times: the better-scoring model is then recommended, not merely ranked first
        slow["ttft_s"], quick["ttft_s"] = 0.6, 9.0
        self.assertEqual(engine.recommended(report)["model"], slow["model"])

    def test_recommendation_ignores_flaky_models(self):
        flaky = {"model": "minimaxai/minimax-m3.1-flash", "status": "flaky", "ok_attempts": 1,
                 "attempts": 2, "ttft_s": 0.3, "trials": []}
        steady = {"model": "nvidia/nemotron-3-ultra-550b-a55b", "status": "ok",
                  "ok_attempts": 2, "attempts": 2, "ttft_s": 0.6, "trials": []}
        report = {"models": {flaky["model"]: flaky, steady["model"]: steady}}
        self.assertEqual(engine.recommended(report)["model"], steady["model"])

    def test_nothing_healthy_means_no_recommendation(self):
        report = {"models": {"a/model-30b": {"model": "a/model-30b", "status": "stall",
                                             "ok_attempts": 0, "attempts": 2, "ttft_s": None,
                                             "trials": []}}}
        self.assertIsNone(engine.recommended(report))
        self.assertIn("nothing", render.best_line(report))

    def test_best_line_says_so_when_nothing_is_healthy(self):
        report = _fake_report({"a/model-30b": "stall"})
        self.assertIn("nothing", render.best_line(report))


class MedianTests(unittest.TestCase):
    def test_median_odd_even_and_empty(self):
        self.assertEqual(engine.median([0.5, 0.1, 0.3]), 0.3)
        self.assertEqual(engine.median([0.4, 0.2]), 0.3)
        self.assertIsNone(engine.median([]))
        self.assertIsNone(engine.median(None))
        self.assertEqual(engine.median([1.0, None, 3.0]), 2.0)

    def test_one_stall_does_not_become_the_speed(self):
        # two fast answers and a 20 s stall: the median must report the fast ones
        self.assertEqual(engine.median([0.3, 0.3, 20.0]), 0.3)


class SpeedTests(unittest.TestCase):
    def _report(self):
        return {"attempts": 3, "models": {
            "slow/model-30b": {"model": "slow/model-30b", "status": "ok", "ok_attempts": 3,
                               "attempts": 3, "ttft_s": 2.0, "ttft_median_s": 2.0,
                               "total_s": 2.6, "trials": ["ok:1.9", "ok:2.0", "ok:2.1"]},
            "fast/model-30b": {"model": "fast/model-30b", "status": "ok", "ok_attempts": 2,
                               "attempts": 3, "ttft_s": 0.2, "ttft_median_s": 0.25,
                               "total_s": 0.9, "trials": ["ok:0.2", "ok:0.3", "stall:25.0"]},
            "dead/model-30b": {"model": "dead/model-30b", "status": "stall", "ok_attempts": 0,
                               "attempts": 3, "ttft_s": None, "ttft_median_s": None,
                               "total_s": None, "trials": ["stall:25.0"]},
        }}

    def test_speed_is_ordered_fastest_first_with_both_timings(self):
        text = render.render_speed(self._report())
        fast = next(line for line in text.splitlines() if line.startswith("1. "))
        self.assertIn("model-30b", fast)
        self.assertNotIn("fast/model-30b", fast)   # org prefix is dropped in the list
        self.assertIn("0.25s", fast)
        self.assertIn("0.9s to finish", fast)
        self.assertIn("0.2 / 0.3", fast)
        self.assertIn("only 2/3 answered", fast)

    def test_unmeasured_models_are_listed_separately_not_silently_dropped(self):
        text = render.render_speed(self._report())
        self.assertIn("no measurement", text)
        self.assertIn("dead/model-30b", text.split("no measurement")[1])

    def test_speed_with_nothing_answering_says_so(self):
        report = {"attempts": 3, "models": {"a/model-30b": {
            "model": "a/model-30b", "status": "stall", "ok_attempts": 0, "attempts": 3,
            "ttft_median_s": None, "trials": ["stall:25"]}}}
        self.assertIn("nothing to time", render.render_speed(report))


class CredentialTests(unittest.TestCase):
    """A revoked key must never be reported as dead models — and never written into their history."""

    def test_models_endpoint_401_is_a_credential_verdict(self):
        err = http_error(401, {"title": "Authentication failed", "detail": "invalid api key"})
        with mock.patch.object(engine, "_urlopen", side_effect=opener([err])):
            status = engine.key_status("https://x/v1", "bad")
        self.assertFalse(status["ok"])
        self.assertEqual(status["code"], 401)
        self.assertIn("invalid api key", status["detail"])

    def test_other_statuses_are_not_credential_verdicts(self):
        err = http_error(500, {"detail": "boom"})
        with mock.patch.object(engine, "_urlopen", side_effect=opener([err])):
            self.assertTrue(engine.key_status("https://x/v1", "k")["ok"])

    def test_network_down_is_not_a_revoked_key(self):
        with mock.patch.object(engine, "_urlopen",
                               side_effect=opener([socket.gaierror("no dns")])):
            self.assertTrue(engine.key_status("https://x/v1", "k")["ok"])

    def test_revoked_key_preserves_model_history_and_alerts_once(self):
        with tempfile.TemporaryDirectory() as directory:
            good = {"run_at": "2026-10-30T10:00:00+00:00", "models": {
                "a/model-30b": {"model": "a/model-30b", "status": "ok", "ok_attempts": 2,
                                "attempts": 2, "severity": "ok", "alerted": False}}}
            engine.save_state(good, directory)

            report = engine._credential_report(
                "https://x/v1", {"ok": False, "code": 401, "detail": "invalid api key"}, directory)

            self.assertTrue(report.get("credential_failure"))
            self.assertEqual(report["last_good_run_at"], "2026-10-30T10:00:00+00:00")
            # the model verdicts survive untouched — nothing here is a statement about a model
            self.assertEqual(report["models"]["a/model-30b"]["severity"], "ok")
            self.assertEqual(report["models"]["a/model-30b"]["status"], "ok")
            self.assertEqual(len(report["changes"]), 1)
            self.assertIn("KEY REJECTED", report["changes"][0])
            self.assertIn("NOT probed", report["changes"][0])

            # a second run inside the re-alert window stays quiet, and keeps models intact
            again = engine._credential_report(
                "https://x/v1", {"ok": False, "code": 401, "detail": "invalid api key"}, directory)
            self.assertEqual(again["changes"], [])
            self.assertEqual(again["models"]["a/model-30b"]["status"], "ok")

    def test_unanimous_401_is_a_credential_verdict(self):
        results = [{"status": "auth_error", "code": 401, "detail": "Authorization failed"},
                   {"status": "auth_error", "code": 401, "detail": "Authorization failed"}]
        verdict = engine.credential_failure(results)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["code"], 401)

    def test_one_rejecting_model_is_not_a_credential_verdict(self):
        # a single 403 can be model-specific entitlement; only unanimity indicts the key
        results = [{"status": "auth_error", "code": 403, "detail": "not entitled"},
                   {"status": "ok", "code": None, "detail": ""}]
        self.assertIsNone(engine.credential_failure(results))
        self.assertIsNone(engine.credential_failure([]))

    def test_banner_and_cached_view(self):
        report = {"run_at": "2026-10-30T11:00:00+00:00",
                  "last_good_run_at": "2026-10-30T10:00:00+00:00",
                  "credential_failure": {"code": 401, "detail": "invalid api key"},
                  "models": {"a/model-30b": {"model": "a/model-30b", "status": "ok",
                                             "ok_attempts": 2, "attempts": 2, "ttft_s": 0.4}}}
        text = render.render_cached(report)
        self.assertIn("🔑 NVIDIA key rejected — HTTP 401", text)
        self.assertIn("invalid api key", text)
        self.assertIn("NVIDIA_API_KEY", text)
        self.assertIn("last good scan", text)
        # the models are still shown, and none of them is claimed to be dead
        self.assertIn("a", text)
        self.assertNotIn("✗", text)
        self.assertIn("cannot measure", render.render_speed(report))


class BrandTests(unittest.TestCase):
    """Every view names the provider. 'NIM' alone is a crate, a compiler, and a dozen other things."""

    def test_every_view_opens_with_the_brand(self):
        report = _fake_report({"nvidia/nemotron-3-ultra-550b-a55b": "ok", "a/x": "stall"})
        report["catalog"] = ["a/x"]
        report["catalog_seen"] = {"a/x": {"first_seen": "2026-10-30T00:00:00+00:00"}}
        report["configured_in"] = {"nvidia/nemotron-3-ultra-550b-a55b": ["fallback_providers[0]"]}
        views = {
            "cached": render.render_cached(report),
            "probe": render.render_probe(report),
            "speed": render.render_speed(report),
            "catalog": render.render_catalog(report),
            "dead": render.render_dead(report),
            "added": render.render_added(report),
            "cfg": render.render_cfg({"a/x": ["fallback_providers[0]"]}, "~/.hermes/config.yaml"),
        }
        for name, text in views.items():
            first = text.splitlines()[0]
            self.assertIn("NVIDIA NIM", first, f"{name} view lost the brand: {first!r}")

    def test_empty_state_and_error_paths_keep_it_too(self):
        self.assertIn("NVIDIA NIM", render.render_cached({}))
        self.assertIn("NVIDIA NIM", render.render_cached({"error": "no key"}))
        self.assertIn("NVIDIA NIM", render.render_speed({"error": "no key"}))


class CatalogTests(unittest.TestCase):
    def test_fetch_failure_returns_empty(self):
        with mock.patch.object(engine, "_urlopen", side_effect=socket.timeout("no")):
            self.assertEqual(engine.catalog("https://x/v1", "k"), [])

    def test_fetch_parses_ids(self):
        body = json.dumps({"data": [{"id": "b/model"}, {"id": "a/model"}]}).encode()
        with mock.patch.object(engine, "_urlopen", lambda *a, **k: FakeResponse(body)):
            self.assertEqual(engine.catalog("https://x/v1", "k"), ["a/model", "b/model"])

    def test_merge_seen_keeps_first_sighting(self):
        previous = {"m": {"first_seen": "2026-01-01T00:00:00+00:00", "last_seen": "2026-01-01T00:00:00+00:00"}}
        merged = engine.merge_seen(previous, ["m", "new"])
        self.assertEqual(merged["m"]["first_seen"], "2026-01-01T00:00:00+00:00")
        self.assertNotEqual(merged["m"]["last_seen"], "2026-01-01T00:00:00+00:00")
        self.assertIn("new", merged)

    def test_merge_seen_keeps_history_of_the_gone(self):
        merged = engine.merge_seen({"gone": {"first_seen": "x", "last_seen": "y"}}, ["live"])
        self.assertIn("gone", merged)
        self.assertEqual(engine.removed_ids({"catalog": ["live"], "catalog_seen": merged}), ["gone"])

    def test_failed_fetch_never_reads_as_mass_removal(self):
        report = {"catalog": ["a"], "catalog_fetch_ok": False,
                  "catalog_seen": {"a": {}, "b": {}}}
        self.assertEqual(engine.removed_ids(report), [])

    def test_added_ids_respects_the_window(self):
        report = {"catalog": ["fresh", "old"],
                  "catalog_fetch_ok": True,
                  "catalog_seen": {
                      "fresh": {"first_seen": "2999-01-01T00:00:00+00:00"},
                      "old": {"first_seen": "2000-01-01T00:00:00+00:00"},
                  }}
        self.assertEqual(engine.added_ids(report, days=7), ["fresh"])


class ConfigScanTests(unittest.TestCase):
    def test_relative_imports_are_safe_to_call(self):
        # A stray absolute import would break the loader's force-reload eviction.
        source = (Path(__file__).resolve().parents[1] / "__init__.py").read_text()
        self.assertIn("from .nim.commands import", source)

    def test_scan_reads_fallback_and_delegation_paths(self):
        config = """
model:
  default: deepseek-flash
fallback_providers:
  - model: openai/gpt-oss-20b
    provider: nvidia
delegation:
  backends:
    model_map:
      options:
        - name: free
          provider: nvidia
          model: nvidia/nemotron-3.5-lightning-30b-a3b
moa:
  reference_models:
    - provider: deepseek
      model: deepseek-v4-flash
    - provider: nvidia
      model: nvidia/nemotron-3-ultra-550b-a55b
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(config)
            found = engine.configured_models(path)
        self.assertEqual(sorted(found), ["nvidia/nemotron-3-ultra-550b-a55b",
                                         "nvidia/nemotron-3.5-lightning-30b-a3b",
                                         "openai/gpt-oss-20b"])
        self.assertTrue(found["openai/gpt-oss-20b"][0].startswith("fallback_providers[0]"))

    def test_shortlist_has_a_default_so_candidates_are_surfaced(self):
        from nim import settings as settings_module

        with mock.patch.dict(os.environ, {"NIM_SHORTLIST": ""}, clear=False):
            self.assertTrue(settings_module.settings().shortlist_models())

    def test_fallback_scanner_without_yaml(self):
        text = ("fallback_providers:\n  - model: nvidia/a\n    provider: nvidia\n"
                "  - model: z/not-nvidia\n    provider: other\n")
        found = engine._configured_models_without_yaml(text)
        self.assertEqual(sorted(found), ["nvidia/a"])


def _fake_report(statuses: dict[str, str], *, catalog=("a", "b"), fetch_ok=True,
                 changes=None) -> dict:
    models = {}
    for model, status in statuses.items():
        models[model] = {"model": model, "status": status, "ok_attempts": 2, "attempts": 2,
                         "ttft_s": 0.5 if status == "ok" else None, "trials": ["ok:0.5"] * 2}
    return {"run_at": engine._now_iso(), "attempts": 2, "models": models,
            "configured_in": {model: ["fallback_providers[0]"] for model in statuses},
            "catalog": list(catalog), "catalog_fetch_ok": fetch_ok,
            "catalog_seen": {model: {"first_seen": engine._now_iso()} for model in catalog},
            "changes": changes or []}


class RenderTests(unittest.TestCase):
    def test_cached_names_models_and_age(self):
        text = render.render_cached(_fake_report({"a": "ok", "b": "stall"}))
        self.assertIn("★ a — 0.5s", text)
        self.assertIn("✗ b — stalled", text)
        self.assertIn("2 models", text)

    def test_a_stale_scan_says_so_in_the_header(self):
        from datetime import datetime, timedelta, timezone

        old = (datetime.now(timezone.utc) - timedelta(hours=9)).isoformat(timespec="seconds")
        report = _fake_report({"a": "ok"})
        report["run_at"] = old
        text = render.render_cached(report)
        self.assertIn("⏳ stale", text.splitlines()[0])
        self.assertIn("/nim probe to measure now", text)

        fresh = _fake_report({"a": "ok"})
        fresh["run_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.assertNotIn("stale", render.render_cached(fresh).splitlines()[0])

    def test_cached_without_state_says_so(self):
        self.assertIn("no scan yet", render.render_cached({}))

    def test_error_report_is_rendered(self):
        self.assertIn("NVIDIA_API_KEY", render.render_cached({"error": engine.key_hint()}))

    def test_paths_become_roles(self):
        self.assertEqual(render.short_where(["fallback_providers[2]"]), "fallback 3")
        self.assertEqual(render.short_where(["delegation.backends.model_map.options[2]"]),
                         "free route")

    def test_dead_and_added(self):
        report = _fake_report({"a": "ok", "b": "not_provisioned"})
        self.assertIn("b", render.render_dead(report))
        # the fixture seeds catalog_seen with "now", so everything it knows is genuinely an arrival
        self.assertIn("first seen", render.render_added(report))
        old = _fake_report({"a": "ok"})
        old["catalog_seen"] = {"a": {"first_seen": "2000-01-01T00:00:00+00:00"}}
        self.assertIn("no new models", render.render_added(old))

    def test_long_lists_are_truncated(self):
        statuses = {f"m{i}": "ok" for i in range(40)}
        text = render.render_cached(_fake_report(statuses, catalog=tuple(statuses)))
        self.assertIn("more (see /nim catalog)", text)


class WatchTests(unittest.TestCase):
    def test_silent_without_changes(self):
        self.assertEqual(watch.render_watch(_fake_report({"a": "ok"})), "")

    def test_speaks_on_change(self):
        text = watch.render_watch(_fake_report({"a": "stall"}, changes=["DEAD     a: STALL"]))
        self.assertIn("NIM watchdog", text)
        self.assertIn("DEAD     a", text)
        self.assertIn("Not usable", text)

    def test_verbose_always_speaks(self):
        self.assertIn("NIM watchdog", watch.render_watch(_fake_report({"a": "ok"}), verbose=True))


class ScanTests(unittest.TestCase):
    """The scan drives hysteresis and history; the transport is stubbed out."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state_dir = self.tmp.name
        self.config = Path(self.state_dir) / "config.yaml"
        self.config.write_text(
            "fallback_providers:\n  - model: nvidia/a\n    provider: nvidia\n"
            "  - model: nvidia/b\n    provider: nvidia\n")
        patcher = mock.patch.dict(os.environ, {"NIM_STATE_DIR": self.state_dir}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def _scan(self, statuses: dict[str, str], catalog=("nvidia/a", "nvidia/b")) -> dict:
        def fake_probe(_base, model, _key, _ttft, attempts, _stagger):
            status = statuses.get(model, "ok")
            return {"model": model, "status": status, "ok_attempts": 1,
                    "attempts": max(1, attempts), "ttft_s": 0.4 if status == "ok" else None,
                    "trials": ["ok:0.4"], "detail": "", "checked_at": engine._now_iso()}

        with mock.patch.object(engine, "probe", side_effect=fake_probe), \
                mock.patch.object(engine, "catalog", return_value=list(catalog)):
            return engine.scan(ttft=1, attempts=1, stagger=0, key="k",
                               config_path=self.config, configured_dir=self.state_dir)

    def test_first_scan_reports_nothing_even_when_broken(self):
        report = self._scan({"nvidia/a": "stall"})
        self.assertEqual(report["changes"], [])
        self.assertEqual(report["models"]["nvidia/a"]["severity"], "dead")

    def test_second_consecutive_bad_scan_reports_once(self):
        self._scan({"nvidia/a": "stall"})
        report = self._scan({"nvidia/a": "stall"})
        self.assertTrue(any(change.startswith("DEAD") for change in report["changes"]))
        third = self._scan({"nvidia/a": "stall"})
        self.assertEqual(third["changes"], [])

    def test_recovery_is_reported_and_rearms(self):
        self._scan({"nvidia/a": "stall"})
        self._scan({"nvidia/a": "stall"})
        recovered = self._scan({})
        self.assertTrue(any(change.startswith("RECOV") for change in recovered["changes"]))
        broke_again = self._scan({"nvidia/a": "stall"})
        self.assertEqual(broke_again["changes"], [])  # needs confirming again

    def test_state_round_trips(self):
        self._scan({})
        saved = engine.load_state(self.state_dir)
        self.assertEqual(saved["version"], engine.STATE_VERSION)
        self.assertIn("nvidia/a", saved["models"])
        self.assertIn("nvidia/a", saved["catalog_seen"])

    def test_failed_catalog_fetch_does_not_announce_removals(self):
        self._scan({})
        report = self._scan({}, catalog=())
        self.assertFalse(report["catalog_fetch_ok"])
        self.assertEqual(report["removed"], [])
        self.assertFalse([c for c in report["changes"] if c.startswith("GONE")])

    def test_removal_is_announced_with_config_flag(self):
        self._scan({})
        report = self._scan({}, catalog=("nvidia/a",))
        gone = [change for change in report["changes"] if change.startswith("GONE")]
        self.assertEqual(len(gone), 1)
        self.assertIn("CONFIG STILL USES IT", gone[0])

    def test_arrival_is_probed_before_being_announced(self):
        self._scan({})
        report = self._scan({}, catalog=("nvidia/a", "nvidia/b", "brand/new"))
        announced = [change for change in report["changes"] if change.startswith("NEW MODEL")]
        self.assertEqual(len(announced), 1)
        self.assertIn("brand/new", announced[0])
        self.assertIn("ok 0.4s", announced[0])

    def test_missing_key_is_an_error_not_a_crash(self):
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": ""}, clear=False), \
                mock.patch.object(engine, "api_key", return_value=""):
            report = engine.scan(key=None, config_path=self.config,
                                 configured_dir=self.state_dir)
        self.assertIn("NVIDIA_API_KEY", report["error"])


if __name__ == "__main__":
    unittest.main()
