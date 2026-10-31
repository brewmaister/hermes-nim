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

from nim import commands, engine, i18n, render, watch  # noqa: E402


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
        first_model = next(line for line in text.splitlines() if line.startswith("⭐"))
        # minimax carries a published AA index (29) and nemotron a lower one (23), so the published
        # score outranks the bigger parameter count — this is the case the size prior got wrong.
        self.assertIn("minimax-m3.1-flash", first_model)
        self.assertNotIn("minimaxai/", first_model)
        self.assertIn("⭐️ recommended minimaxai/minimax-m3.1-flash", text)
        # model rows stay narrow enough not to wrap twice on a phone (the footer may run longer:
        # it is the one line carrying a full id you would paste into config.yaml)
        rows = [line for line in text.splitlines()
                if line.startswith(("⭐", "· ", "⚠", "🔴"))
                and "recommended" not in line]
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
        self.assertNotIn("🔴", text)
        self.assertIn("cannot measure", render.render_speed(report))


class FakeHandle:
    def __init__(self, ok=True, fail_edit=False):
        self.ok = ok
        self.edited = []
        self._fail_edit = fail_edit

    def edit(self, text):
        if self._fail_edit:
            raise RuntimeError("surface cannot edit")
        self.edited.append(text)
        return {"ok": True}


class FakeCtx:
    def __init__(self, handle=None, raises=False):
        self.handle = handle
        self.raises = raises
        self.posted = []

    def post_message(self, text):
        if self.raises:
            raise RuntimeError("no conversation bound")
        self.posted.append(text)
        return self.handle


class LiveDefaultTests(unittest.TestCase):
    """/nim measures by default; the cache is one word away and never slower than it was."""

    def _settings(self, live=True, grace=20.0, survey=False):
        """survey=False on purpose: these tests are about the config-only door.

        Leaving it True would route dispatch into do_survey() and the suite would hit NVIDIA's API —
        a test that needs the network is not a test, it is a flaky reminder.
        """
        from nim import settings as settings_mod

        return settings_mod.Settings(live_default=live, live_grace_seconds=grace,
                                     survey_catalog=survey)

    def _fresh(self, **overrides):
        from datetime import datetime, timezone

        model = {"model": "a/x-30b", "status": "ok", "ok_attempts": 2, "attempts": 2,
                 "ttft_s": 0.4, "severity": "ok", "trials": ["ok:0.4", "ok:0.4"]}
        model.update(overrides)
        return {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "attempts": 2, "models": {"a/x-30b": model}}

    def test_default_verb_probes_when_the_cache_is_old(self):
        stale = self._fresh()
        stale["run_at"] = "2026-10-30T00:00:00+00:00"
        with mock.patch.object(commands.settings, "settings", return_value=self._settings()), \
                mock.patch.object(commands.engine, "load_state", return_value=stale), \
                mock.patch.object(commands, "do_scan", return_value=self._fresh()) as scan:
            text, code = commands.dispatch("", [])
        scan.assert_called_once()
        self.assertEqual(code, 0)
        self.assertIn("just now", text)

    def test_a_scan_younger_than_the_grace_is_reused(self):
        with mock.patch.object(commands.settings, "settings", return_value=self._settings()), \
                mock.patch.object(commands.engine, "load_state", return_value=self._fresh()), \
                mock.patch.object(commands, "do_scan", side_effect=AssertionError("probed again")):
            text, _ = commands.dispatch("", [])
        self.assertIn("x-30b", text)

    def test_cached_verb_never_touches_the_network(self):
        with mock.patch.object(commands.settings, "settings", return_value=self._settings()), \
                mock.patch.object(commands.engine, "load_state", return_value=self._fresh()), \
                mock.patch.object(commands, "do_scan", side_effect=AssertionError("probed")):
            text, _ = commands.dispatch("cached", [])
        self.assertIn("ago", text.splitlines()[0])

    def test_the_default_door_is_the_survey_when_that_is_configured(self):
        fresh = self._fresh()
        fresh["catalog"] = ["a/x-30b"]
        fresh["survey"] = {"at": fresh["run_at"], "listed": 85, "measured": 85, "online": 13}
        with mock.patch.object(commands.settings, "settings",
                               return_value=self._settings(survey=True)), \
                mock.patch.object(commands.engine, "load_state",
                                  return_value={"run_at": "2026-10-30T00:00:00+00:00"}), \
                mock.patch.object(commands, "do_survey", return_value=fresh) as survey:
            text, code = commands.dispatch("", [])
        survey.assert_called_once()
        self.assertEqual(code, 0)
        self.assertIn("🗒1 listed", text)      # the stats line proves the survey path ran
        self.assertIn("⌛️", text)

    def test_live_default_off_restores_the_old_behaviour(self):
        with mock.patch.object(commands.settings,
                               "settings", return_value=self._settings(live=False)), \
                mock.patch.object(commands.engine, "load_state", return_value=self._fresh()), \
                mock.patch.object(commands, "do_scan", side_effect=AssertionError("probed")):
            text, _ = commands.dispatch("", [])
        self.assertIn("ago", text.splitlines()[0])

    def test_will_measure_matches_the_verbs(self):
        with mock.patch.object(commands.settings, "settings", return_value=self._settings()):
            self.assertTrue(commands.will_measure(""))
            self.assertTrue(commands.will_measure("probe"))
            self.assertTrue(commands.will_measure("speed"))
            self.assertFalse(commands.will_measure("cached"))
            self.assertFalse(commands.will_measure("catalog"))
        with mock.patch.object(commands.settings,
                               "settings", return_value=self._settings(live=False)):
            self.assertFalse(commands.will_measure(""))


class ProgressMessageTests(unittest.TestCase):
    """A 20-40 s silence reads as a broken bot: say so first, then edit that message."""

    def test_progress_note_is_replaced_by_the_report(self):
        handle = FakeHandle()
        ctx = FakeCtx(handle)
        with mock.patch.object(commands.settings, "settings", return_value=LiveDefaultTests()._settings()), \
                mock.patch.object(commands, "slash_command", return_value="🧠 NVIDIA NIM · just now"):
            result = commands.slash_command_with_progress(ctx, "")
        self.assertIsNone(result)                      # delivered in place, nothing extra sent
        self.assertEqual(len(ctx.posted), 1)
        self.assertIn("measuring", ctx.posted[0])
        self.assertEqual(handle.edited, ["🧠 NVIDIA NIM · just now"])

    def test_instant_verbs_post_nothing(self):
        ctx = FakeCtx(FakeHandle())
        with mock.patch.object(commands.settings, "settings", return_value=LiveDefaultTests()._settings()), \
                mock.patch.object(commands, "slash_command", return_value="cached text"):
            self.assertEqual(commands.slash_command_with_progress(ctx, "cached"), "cached text")
        self.assertEqual(ctx.posted, [])

    def test_a_refused_post_still_returns_the_text(self):
        ctx = FakeCtx(FakeHandle(ok=False))
        with mock.patch.object(commands.settings, "settings", return_value=LiveDefaultTests()._settings()), \
                mock.patch.object(commands, "slash_command", return_value="report"):
            self.assertEqual(commands.slash_command_with_progress(ctx, ""), "report")

    def test_a_broken_editing_surface_still_returns_the_text(self):
        ctx = FakeCtx(FakeHandle(fail_edit=True))
        with mock.patch.object(commands.settings, "settings", return_value=LiveDefaultTests()._settings()), \
                mock.patch.object(commands, "slash_command", return_value="report"):
            self.assertEqual(commands.slash_command_with_progress(ctx, ""), "report")

    def test_no_conversation_never_breaks_the_command(self):
        ctx = FakeCtx(raises=True)
        with mock.patch.object(commands.settings, "settings", return_value=LiveDefaultTests()._settings()), \
                mock.patch.object(commands, "slash_command", return_value="report"):
            self.assertEqual(commands.slash_command_with_progress(ctx, ""), "report")


class EmojiFormatTests(unittest.TestCase):
    """One glyph per fact: 🧠 published score, 📐 parameter count, ⚡ measured time."""

    def _report(self):
        return _fake_report({}) | {"models": {
            "minimaxai/minimax-m3.1-flash": {"model": "minimaxai/minimax-m3.1-flash",
                                             "status": "ok", "ok_attempts": 2, "attempts": 2,
                                             "ttft_s": 0.41},
            "nvidia/nemotron-3.5-lightning-30b-a3b": {
                "model": "nvidia/nemotron-3.5-lightning-30b-a3b", "status": "ok",
                "ok_attempts": 2, "attempts": 2, "ttft_s": 0.33},
            "vendor/mystery-model": {"model": "vendor/mystery-model", "status": "stall",
                                     "ok_attempts": 0, "attempts": 2, "ttft_s": None},
        }}

    def test_marks_carry_the_meaning(self):
        text = render.render_cached(self._report())
        self.assertIn("🧠29 family", text)          # published index, inherited family score
        self.assertIn("⚡0.41s", text)              # measured first byte
        # a model with no published score gets no 🧠 — the glyph is never filled in with a guess
        lightning = next(line for line in text.splitlines() if "lightning" in line)
        self.assertNotIn("🧠", lightning)
        self.assertIn("⚡0.33s", lightning)
        # a stalled model has no duration, so it carries no ⚡
        stalled = next(line for line in text.splitlines() if line.startswith("🔴"))
        self.assertNotIn("⚡", stalled)
        self.assertIn("stalled", stalled)

    def test_a_size_the_id_already_states_is_never_repeated(self):
        line = render._line({"model": "nvidia/nemotron-3.5-lightning-30b-a3b", "status": "ok",
                             "ok_attempts": 1, "attempts": 1, "ttft_s": 0.3}, {})
        self.assertNotIn("30B", line)
        self.assertIn("30b", line)   # it is in the name, which is where a reader looks for it

    def test_the_two_counts_say_which_is_which(self):
        """"5 models" over "85 tracked" reads as "scanned 85, five answered". It must not."""
        survey = {"at": engine._now_iso(), "listed": 2, "measured": 2, "online": 1}
        report = _fake_report({"a/one": "ok", "a/two": "stall"}, catalog=("a/one", "a/two"),
                              survey=survey)
        text = render.render_cached(report)
        # one line, three facts: how many you point at, how many of those the scan measured, and how
        # many of everything listed answers — so "2 models" cannot be read as "only 2 exist"
        self.assertIn("your 2 models", text)
        self.assertIn("1/2 listed answer", text)
        self.assertIn("board: 1 of 2 measured models answer", text)

    def test_an_unsurveyed_catalog_does_not_pretend_to_a_verdict(self):
        report = _fake_report({"a/one": "ok"}, catalog=("a/one", "other/x", "other/y"))
        text = render.render_cached(report)
        self.assertIn("board: 3 listed · not surveyed yet", text)

    def test_the_star_row_says_recommended_instead_of_its_config_role(self):
        report = self._report()
        report["configured_in"] = {"minimaxai/minimax-m3.1-flash": ["shortlist"]}
        star = next(line for line in render.render_cached(report).splitlines()
                    if line.startswith("⭐"))
        self.assertIn("recommended", star)
        self.assertNotIn("shortlist", star)   # one row, one claim

    def test_the_star_row_is_spanish_too(self):
        from nim import i18n

        report = self._report()
        with mock.patch.dict(os.environ, {"HERMES_LANGUAGE": "es"}):
            i18n.reset_cache()
            star = next(line for line in render.render_cached(report).splitlines()
                        if line.startswith("⭐"))
        self.assertIn("recomendado", star)

    def test_the_title_is_plain_text_and_has_no_glyph(self):
        """Plugin output is delivered as plain text: `**bold**` reaches the reader as asterisks."""
        first = render.render_cached(self._report()).splitlines()[0]
        self.assertTrue(first.startswith(i18n.BOLD_BRAND), first)
        self.assertNotIn("🧠", first)

    def test_no_view_emits_markdown(self):
        """Every door is plain text. A stray `**` or backtick is a bug the reader sees literally."""
        report = self._report()
        views = [render.render_cached(report), render.render_probe(report),
                 render.render_board(report), render.render_catalog(report),
                 render.render_dead(report), render.render_added(report),
                 render.render_speed(report), commands.usage()]
        from nim import settings as settings_module
        views.append(settings_module.describe())
        for text in views:
            self.assertNotIn("**", text)
            self.assertNotIn("`", text)

    def test_plain_strips_the_emphasis_for_terminals(self):
        text = render.render_cached(self._report())
        flat = render.plain(text)
        self.assertNotIn("**", flat)
        self.assertIn(i18n.BOLD_BRAND + " · ", flat)
        # the glyphs survive: they are content, not decoration
        self.assertIn("🧠29", flat)


class CapabilityTests(unittest.TestCase):
    """The chat/utility split is a coded naming rule, not somebody's taste — and it is testable."""

    def test_utility_models_are_named_utility_models(self):
        for model in ("meta/llama-guard-4-12b", "nvidia/nemotron-safety-guard-8b-v3",
                      "nvidia/nemotron-parse", "nvidia/ising-calibration-1.5-31b",
                      "ai-synthetic-video-detector", "nvidia/llama-3.2-90b-vision-instruct",
                      "baai/bge-m3-embed"):
            self.assertEqual(engine.capability(model), "utility", model)

    def test_everything_else_is_treated_as_chat(self):
        for model in ("minimaxai/minimax-m3.1-flash", "nvidia/nemotron-3-ultra-550b-a55b",
                      "openai/gpt-oss-20b", "meta/muse-glimmer-30b", "poolside/laguna-xs-2.1"):
            self.assertEqual(engine.capability(model), "chat", model)

    def test_the_board_splits_online_models_and_counts_the_dead(self):
        survey = {"at": engine._now_iso(), "listed": 4, "measured": 4, "online": 2}
        ids = ("minimaxai/minimax-m3.1-flash", "nvidia/llama-guard-4-12b", "yi-large",
               "nvidia/nemotron-parse")
        report = _fake_report({"minimaxai/minimax-m3.1-flash": "ok",
                               "nvidia/llama-guard-4-12b": "ok",
                               "yi-large": "not_provisioned",
                               "nvidia/nemotron-parse": "not_provisioned"},
                              catalog=ids, survey=survey)
        text = render.render_board(report)
        self.assertIn("🟢 1 chat · 🛠 1 · 🔴 2 offline", text)      # stats line
        self.assertEqual(text.splitlines()[0], "NVIDIA NIM")        # title on its own line
        self.assertNotIn("🏢", text)                                 # vendors live in /nim vendor
        self.assertIn("💬 chat models", text)
        self.assertIn("minimax-m3.1-flash", text)
        self.assertIn("🛠 utility models", text)
        self.assertIn("llama-guard-4-12b", text)
        self.assertIn("🔴 2 offline", text)
        self.assertIn("2 not provisioned", text)

    def test_every_utility_marker_has_a_label_in_every_language(self):
        """A missing label would print `(util.parse)` at the reader — worse than no label."""
        from nim import i18n

        for marker in engine.UTILITY_MARKERS:
            for lang in ("en", "es"):
                label = i18n.t(f"util.{marker}", lang=lang)
                self.assertNotEqual(label, f"util.{marker}", f"{marker} missing in {lang}")

    def test_utility_rows_say_what_the_model_is(self):
        survey = {"at": engine._now_iso(), "listed": 3, "measured": 3, "online": 3}
        ids = ("meta/llama-guard-4-12b", "nvidia/nemotron-parse-2.0", "openai/gpt-oss-20b")
        report = _fake_report({model: "ok" for model in ids}, catalog=ids, survey=survey)
        text = render.render_board(report)
        self.assertIn("· llama-guard-4-12b (safety classifier)", text)
        self.assertIn("· nemotron-parse-2.0 (document parsing)", text)
        self.assertNotIn("gpt-oss-20b (", text)      # a chat model carries no utility label

    def test_the_board_survives_a_report_with_no_survey(self):
        report = _fake_report({"a/one": "ok"}, catalog=("a/one",))
        self.assertIn("never surveyed", render.render_board(report))


class ResponseDataTests(unittest.TestCase):
    """The answer carries more than a status: tokens, writing speed, backend, request id."""

    def test_the_probe_path_follows_the_model_kind(self):
        self.assertEqual(engine.endpoint_for("nvidia/embed-qa-4"), "embeddings")
        self.assertEqual(engine.endpoint_for("baai/bge-m3-rerank"), "embeddings")
        self.assertEqual(engine.endpoint_for("openai/gpt-oss-20b"), "chat/completions")

    def test_probe_keeps_what_the_answer_said_about_itself(self):
        def fake_attempt(_b, model, _k, _ttft, _endpoint=None):
            return {"kind": "ok", "seconds": 0.4, "total": 1.4, "detail": "", "tokens": 32,
                    "tps": 32.0, "tier": "default", "fingerprint": "vllm-abc123456789",
                    "req_id": "3d49dd76-4c14-41a8-bf05-a0a5ea5abfa7"}

        with mock.patch.object(engine, "attempt", side_effect=fake_attempt):
            result = engine.probe("http://x", "a/model", "k", ttft=1, attempts=1)
        self.assertEqual(result["tokens"], 32)
        self.assertEqual(result["tps"], 32.0)
        self.assertEqual(result["tier"], "default")
        self.assertEqual(result["req_id"], "3d49dd76-4c14-41a8-bf05-a0a5ea5abfa7")
        self.assertTrue(result["fingerprint"].startswith("vllm"))

    def test_a_failure_still_carries_its_request_id(self):
        def fake_attempt(_b, _m, _k, _t, _endpoint=None):
            return {"kind": "not_provisioned", "seconds": 0.2, "detail": "nope",
                    "req_id": "abcdef12-3456"}

        with mock.patch.object(engine, "attempt", side_effect=fake_attempt):
            result = engine.probe("http://x", "a/model", "k", ttft=1, attempts=1)
        self.assertEqual(result["req_id"], "abcdef12-3456")

    def test_vendors_have_their_own_view(self):
        survey = {"at": engine._now_iso(), "listed": 3, "measured": 3, "online": 2}
        ids = ("openai/gpt-oss-20b", "nvidia/a", "meta/thing")
        report = _fake_report({"openai/gpt-oss-20b": "ok", "nvidia/a": "ok",
                               "meta/thing": "not_provisioned"}, catalog=ids, survey=survey)
        report["models"]["openai/gpt-oss-20b"].update({"ttft_s": 0.4, "tps": 40.0})
        report["models"]["nvidia/a"].update({"ttft_s": 0.6, "tps": 20.0})
        report["catalog_owners"] = {"openai/gpt-oss-20b": "openai", "nvidia/a": "nvidia",
                                    "meta/thing": "meta"}
        text = render.render_vendors(report)
        self.assertIn("NVIDIA NIM · vendors", text)
        self.assertIn("🏢 openai — 1/1 online · ⚡0.4s · ✍️40t/s", text)
        self.assertIn("🏢 meta — 0/1 online", text)

    def test_the_board_shows_speed_and_tier(self):
        survey = {"at": engine._now_iso(), "listed": 2, "measured": 2, "online": 2}
        report = _fake_report({"openai/gpt-oss-20b": "ok", "nvidia/nemotron-3-ultra-550b-a55b": "ok"},
                              catalog=("openai/gpt-oss-20b",
                                       "nvidia/nemotron-3-ultra-550b-a55b"), survey=survey)
        for verdict in report["models"].values():
            verdict.update({"tokens": 32, "tps": 41.5, "tier": "default",
                            "fingerprint": "vllm-127abc", "req_id": "req-123456789"})
        report["catalog_owners"] = {"openai/gpt-oss-20b": "openai",
                                    "nvidia/nemotron-3-ultra-550b-a55b": "nvidia"}
        text = render.render_board(report)
        self.assertIn("✍️41.5t/s", text)
        self.assertIn("tier default", render.best_line(report))

    def test_a_dead_row_carries_the_id_you_can_quote_to_support(self):
        report = _fake_report({"a/one": "not_provisioned"}, catalog=("a/one",))
        report["models"]["a/one"]["req_id"] = "3d49dd76-4c14"
        self.assertIn("req 3d49dd76", render.render_dead(report))


class NewModelTests(unittest.TestCase):
    """🆕 has to mean "arrived", not "was there on install day"."""

    def test_the_install_baseline_is_never_flagged_new(self):
        def fake_probe(_b, model, _k, _t, attempts, _s):
            return {"model": model, "status": "ok", "ok_attempts": 1, "attempts": attempts,
                    "ttft_s": 0.4, "trials": ["ok:0.4"], "detail": "",
                    "checked_at": engine._now_iso()}

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text("fallback_providers:\n  - model: nvidia/a\n    provider: nvidia\n")
            with mock.patch.object(engine, "probe", side_effect=fake_probe), \
                    mock.patch.object(engine, "catalog_entries", return_value=(["a/one", "a/two"], {})):
                first = engine.scan(ttft=1, attempts=1, stagger=0, key="k", config_path=config,
                                    configured_dir=tmp, survey=True, survey_ttft=1)
        self.assertEqual(render.new_models(first), set())      # baseline: 85 "new" would be a lie
        self.assertNotIn(render.NEW_MARK, render.render_board(first))

    def test_a_model_that_appears_later_is_flagged(self):
        def fake_probe(_b, model, _k, _t, attempts, _s):
            return {"model": model, "status": "ok", "ok_attempts": 1, "attempts": attempts,
                    "ttft_s": 0.4, "trials": ["ok:0.4"], "detail": "",
                    "checked_at": engine._now_iso()}

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text("fallback_providers:\n  - model: nvidia/a\n    provider: nvidia\n")
            with mock.patch.object(engine, "probe", side_effect=fake_probe), \
                    mock.patch.object(engine, "catalog_entries", return_value=(["a/one"], {})):
                engine.scan(ttft=1, attempts=1, stagger=0, key="k", config_path=config,
                            configured_dir=tmp, survey=True, survey_ttft=1)
            with mock.patch.object(engine, "probe", side_effect=fake_probe), \
                    mock.patch.object(engine, "catalog_entries", return_value=(["a/one", "a/brand-new"], {})):
                later = engine.scan(ttft=1, attempts=1, stagger=0, key="k", config_path=config,
                                    configured_dir=tmp, survey=True, survey_ttft=1)
        self.assertEqual(render.new_models(later), {"a/brand-new"})
        board = render.render_board(later)
        self.assertIn(render.NEW_MARK, board)
        self.assertIn("🆕 1 new in 7 days", board)


class LocalizationTests(unittest.TestCase):
    """Every string comes from locales/; a missing key must degrade to English, never to a key name."""

    def setUp(self):
        from nim import i18n

        i18n.reset_cache()

    def tearDown(self):
        from nim import i18n

        i18n.reset_cache()

    def _report(self):
        return _fake_report({"nvidia/nemotron-3-ultra-550b-a55b": "ok", "a/x": "stall"})

    def test_english_is_the_source_and_the_fallback(self):
        with mock.patch.dict(os.environ, {"HERMES_LANGUAGE": "en"}):
            self.assertIn("models", render.render_cached(self._report()))

    def test_spanish_renders_the_same_report_in_spanish(self):
        from nim import i18n

        with mock.patch.dict(os.environ, {"HERMES_LANGUAGE": "es"}):
            i18n.reset_cache()
            text = render.render_cached(self._report())
        self.assertIn("modelos", text)
        self.assertIn("recomendado", text)
        self.assertIn("colgado", text)        # status.stalled
        self.assertIn("respaldo 1", text)     # role.fallback
        self.assertNotIn("recommended", text)
        self.assertNotIn("stalled", text)
        # the /nim dead view has its own wording — inside the same language scope
        from nim import i18n

        with mock.patch.dict(os.environ, {"HERMES_LANGUAGE": "es"}):
            i18n.reset_cache()
            self.assertIn("sin usar", render.render_dead(self._report()))

    def test_help_text_is_translated(self):
        from nim import commands

        from nim import i18n

        with mock.patch.dict(os.environ, {"HERMES_LANGUAGE": "es"}):
            i18n.reset_cache()
            self.assertIn("escaneo en vivo", commands.usage())
        with mock.patch.dict(os.environ, {"HERMES_LANGUAGE": "en"}):
            i18n.reset_cache()
            self.assertIn("live scan, now", commands.usage())

    def test_missing_key_falls_back_to_english_not_to_the_key(self):
        from nim import i18n

        self.assertEqual(i18n.t("does.not.exist", lang="es"), "does.not.exist")
        # a key only English has: English text, in Spanish mode
        i18n._catalog.cache_clear()
        self.assertEqual(i18n.t("speed.none", lang="es"), "Ningún modelo respondió — no hay nada que medir.")

    def test_a_translation_with_a_broken_placeholder_does_not_raise(self):
        from nim import i18n

        # es.yaml's rec.line wants {model}/{evidence}; asking for a key with wrong fields must not blow up
        self.assertIsInstance(i18n.t("rec.line", lang="es"), str)

    def test_no_english_leaks_into_a_spanish_report(self):
        """A translated page with one English word in it is the bug the reader notices first."""
        from nim import commands, i18n, settings

        report = self._report()
        # config keys (shortlist, live_default, state_dir) stay verbatim on purpose: you type those
        # into config.yaml, so translating the label would break the copy-paste.
        english = ("family", "recommended", "fallback", "settings (", "models",
                   "just now", "catalog:", "Unknown argument", "no scan yet")
        with mock.patch.dict(os.environ, {"HERMES_LANGUAGE": "es"}):
            i18n.reset_cache()
            views = [render.render_cached(report), render.render_dead(report),
                     render.render_catalog(report), render.render_added(report),
                     render.render_probe(report), render.render_speed(report),
                     commands.usage(), settings.describe()]
        import re

        for text in views:
            # /v1/models is the API path, not prose — it must survive translation untouched.
            # Config keys (survey_catalog:, ttft_seconds:) are identifiers you type into config.yaml,
            # so they are stripped before the prose check rather than translated.
            scan = text.replace("/v1/models", "/v1/MODELS")
            scan = re.sub(r"\b[a-z][a-z0-9_]*:", "", scan)
            for word in english:
                self.assertNotIn(word, scan, f"'{word}' is English, in Spanish output:\n{text[:200]}")

    def test_the_family_tag_is_translated(self):
        from nim import i18n

        # minimax-m3.1-flash inherits its index from the M3 family, so it carries the tag
        report = _fake_report({"minimaxai/minimax-m3.1-flash": "ok"})
        with mock.patch.dict(os.environ, {"HERMES_LANGUAGE": "es"}):
            i18n.reset_cache()
            self.assertIn("familia", render.render_cached(report))
        with mock.patch.dict(os.environ, {"HERMES_LANGUAGE": "en"}):
            i18n.reset_cache()
            self.assertIn("family", render.render_cached(report))

    def test_every_english_key_exists_in_every_shipped_language(self):
        from nim import i18n

        english = i18n._catalog("en")
        self.assertTrue(english)
        for language in i18n.available():
            catalog = i18n._catalog(language)
            missing = sorted(set(english) - set(catalog))
            self.assertEqual(missing, [], f"{language}.yaml is missing {missing}")

    def test_unknown_language_falls_back_to_english_output(self):
        from nim import i18n

        with mock.patch.dict(os.environ, {"HERMES_LANGUAGE": "tlh"}):
            i18n.reset_cache()
            self.assertIn("models", render.render_cached(self._report()))


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
            self.assertIn(i18n.BOLD_BRAND, first, f"{name} view lost the brand: {first!r}")

    def test_empty_state_and_error_paths_keep_it_too(self):
        self.assertIn(i18n.BOLD_BRAND, render.render_cached({}))
        self.assertIn(i18n.BOLD_BRAND, render.render_cached({"error": "no key"}))
        self.assertIn(i18n.BOLD_BRAND, render.render_speed({"error": "no key"}))


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

    def test_nothing_is_probed_that_you_did_not_point_at(self):
        """The default shortlist is empty: a plugin does not get to spend your latency on its taste."""
        from nim import settings as settings_module

        with mock.patch.dict(os.environ, {"NIM_SHORTLIST": ""}, clear=False), \
                mock.patch.object(settings_module, "settings",
                                  return_value=settings_module.Settings()):
            self.assertEqual(settings_module.settings().shortlist_models(), [])

    def test_the_ctx_less_doors_read_config_yaml(self):
        """`hermes nim` and the cron watchdog have no ctx — they must still honour config.yaml."""
        from nim import settings as settings_module

        file_settings = {"shortlist": "minimaxai/minimax-m3.1-flash", "ttft_seconds": 9,
                         "language": "es"}
        with mock.patch.object(settings_module, "_CTX", None), \
                mock.patch.object(settings_module, "_config_entry_from_file",
                                  return_value=file_settings):
            current = settings_module.settings()
        self.assertEqual(current.shortlist_models(), ["minimaxai/minimax-m3.1-flash"])
        self.assertEqual(current.ttft_seconds, 9)
        self.assertEqual(current.language, "es")

    def test_the_gateway_ctx_still_wins_over_the_file(self):
        from nim import settings as settings_module

        ctx = mock.Mock()
        ctx.get_config.return_value = {"plugins": {"entries": {"hermes-nim": {"settings": {
            "shortlist": "from/ctx"}}}}}
        with mock.patch.object(settings_module, "_CTX", ctx), \
                mock.patch.object(settings_module, "_config_entry_from_file",
                                  return_value={"shortlist": "from/file"}):
            self.assertEqual(settings_module.settings().shortlist_models(), ["from/ctx"])

    def test_a_broken_config_file_falls_back_to_defaults(self):
        from nim import settings as settings_module

        with mock.patch.object(settings_module, "_CTX", None), \
                mock.patch.object(settings_module, "_config_entry_from_file", return_value={}):
            self.assertEqual(settings_module.settings().shortlist_models(), [])

    def test_a_shortlist_you_set_is_probed(self):
        from nim import settings as settings_module

        with mock.patch.dict(os.environ, {"NIM_SHORTLIST": "a/one, b/two"}, clear=False):
            self.assertEqual(settings_module.Settings().with_env().shortlist_models(),
                             ["a/one", "b/two"])

    def test_scan_measures_config_models_only_without_a_shortlist(self):
        """The count in the progress message is 4 for this box because 4 is what config.yaml says."""
        probed: list[str] = []

        def fake_probe(_base, model, _key, _ttft, attempts, _stagger):
            probed.append(model)
            return {"model": model, "status": "ok", "ok_attempts": 1, "attempts": 1,
                    "ttft_s": 0.4, "trials": ["ok:0.4"], "detail": "",
                    "checked_at": engine._now_iso()}

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text("fallback_providers:\n  - model: nvidia/a\n    provider: nvidia\n"
                              "  - model: nvidia/b\n    provider: nvidia\n")
            with mock.patch.object(engine, "probe", side_effect=fake_probe), \
                    mock.patch.object(engine, "catalog_entries", return_value=(["nvidia/a", "nvidia/b"], {})):
                engine.scan(ttft=1, attempts=1, stagger=0, key="k", extra=[],
                            config_path=config, configured_dir=tmp)
        self.assertEqual(sorted(probed), ["nvidia/a", "nvidia/b"])

    def test_fallback_scanner_without_yaml(self):
        text = ("fallback_providers:\n  - model: nvidia/a\n    provider: nvidia\n"
                "  - model: z/not-nvidia\n    provider: other\n")
        found = engine._configured_models_without_yaml(text)
        self.assertEqual(sorted(found), ["nvidia/a"])


def _fake_report(statuses: dict[str, str], *, catalog=("a", "b"), fetch_ok=True,
                 changes=None, survey=None, new=()) -> dict:
    """A report shaped like the real one. Everything in `catalog` is baseline (seeded) unless it is
    listed in `new` — otherwise every fixture would be flagged 🆕 and the marker would mean nothing."""
    models = {}
    for model, status in statuses.items():
        models[model] = {"model": model, "status": status, "ok_attempts": 2, "attempts": 2,
                         "ttft_s": 0.5 if status == "ok" else None, "trials": ["ok:0.5"] * 2}
    return {"run_at": engine._now_iso(), "attempts": 2, "models": models, "survey": survey,
            "configured_in": {model: ["fallback_providers[0]"] for model in statuses},
            "catalog": list(catalog), "catalog_fetch_ok": fetch_ok,
            "catalog_seen": {model: {"first_seen": engine._now_iso(),
                                     "seeded": model not in new} for model in catalog},
            "changes": changes or []}


class RenderTests(unittest.TestCase):
    def test_cached_names_models_and_age(self):
        text = render.render_cached(_fake_report({"a": "ok", "b": "stall"}))
        self.assertIn("⭐️ a — ⚡0.5s", text)
        self.assertIn("🔴 b — stalled", text)
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
        # role words now come from the locale catalog; English says "free" (it used to say
        # "free route" and get shortened later — one name, one place)
        self.assertEqual(render.short_where(["delegation.backends.model_map.options[2]"]), "free")

    def test_dead_and_added(self):
        report = _fake_report({"a": "ok", "b": "not_provisioned"})
        self.assertIn("b", render.render_dead(report))
        # a fixture marked `new` is an arrival; the default baseline is not
        arrival = _fake_report({"a": "ok"}, catalog=("a", "brand/new"), new=("brand/new",))
        self.assertIn("brand/new", render.render_added(arrival))
        self.assertIn("no new models", render.render_added(report))
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
                mock.patch.object(engine, "catalog_entries", return_value=(list(catalog), {})):
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

    def _caps(self, recheck: float) -> dict[str, float]:
        """Run one scan and record the first-byte cap each model was probed with."""
        seen: dict[str, float] = {}

        def fake_probe(_base, model, _key, ttft, attempts, _stagger):
            seen[model] = ttft
            return {"model": model, "status": "stall", "ok_attempts": 0, "attempts": 1,
                    "ttft_s": None, "trials": ["stall:25"], "detail": "",
                    "checked_at": engine._now_iso()}

        with mock.patch.object(engine, "probe", side_effect=fake_probe), \
                mock.patch.object(engine, "catalog_entries", return_value=(["nvidia/a", "nvidia/b"], {})):
            engine.scan(ttft=25, attempts=1, stagger=0, key="k", config_path=self.config,
                        configured_dir=self.state_dir, recheck_ttft=recheck)
        return seen

    def test_a_model_already_dead_is_rechecked_with_the_shorter_cap(self):
        """25 s of silence per dead model per sweep bought nothing; re-confirming a death is cheaper."""
        self.assertEqual(self._caps(recheck=8), {"nvidia/a": 25, "nvidia/b": 25})  # first verdict
        self.assertEqual(self._caps(recheck=8), {"nvidia/a": 8, "nvidia/b": 8})    # already dead
        # a model that answers again still gets the real cap on its next run, so a slow recovery is
        # judged against ttft_seconds rather than the recheck shortcut
        self._scan({})
        self.assertEqual(self._caps(recheck=8), {"nvidia/a": 25, "nvidia/b": 25})

    def test_a_survey_measures_every_listed_model_once(self):
        """The board needs all 85 measured, but only the models you point at may raise an alert."""
        probed: list[tuple[str, int, float]] = []

        def fake_probe(_base, model, _key, ttft, attempts, _stagger):
            probed.append((model, attempts, ttft))
            status = "ok" if model == "nvidia/a" else "http_404"
            return {"model": model, "status": status, "ok_attempts": 1, "attempts": attempts,
                    "ttft_s": 0.4 if status == "ok" else None,
                    "trials": ["ok:0.4"] if status == "ok" else ["http_404"], "detail": "",
                    "checked_at": engine._now_iso()}

        catalog = ["nvidia/a", "nvidia/b", "cloud/one", "cloud/two"]
        with mock.patch.object(engine, "probe", side_effect=fake_probe), \
                mock.patch.object(engine, "catalog_entries", return_value=(list(catalog), {})):
            report = engine.scan(ttft=25, attempts=2, stagger=0, key="k", config_path=self.config,
                                 configured_dir=self.state_dir, survey=True, survey_ttft=10)
        self.assertEqual(sorted(model for model, _, _ in probed), sorted(catalog))
        # catalog-only models: one attempt, the survey cap. Configured ones keep the real policy.
        caps = {model: (attempts, ttft) for model, attempts, ttft in probed}
        self.assertEqual(caps["nvidia/a"], (2, 25))
        self.assertEqual(caps["cloud/one"], (1, 10))
        # nobody in config → no alert may be produced by a survey of strangers
        self.assertEqual(report["changes"], [])

    def test_chat_views_stay_on_the_models_you_point_at_after_a_survey(self):
        """A stored survey holds 85 verdicts; `/nim` must not turn into a catalog dump."""
        def fake_probe(_base, model, _key, _ttft, attempts, _stagger):
            return {"model": model, "status": "ok", "ok_attempts": 1, "attempts": attempts,
                    "ttft_s": 0.4, "trials": ["ok:0.4"], "detail": "",
                    "checked_at": engine._now_iso()}

        with mock.patch.object(engine, "probe", side_effect=fake_probe), \
                mock.patch.object(engine, "catalog_entries",
                                  return_value=(["nvidia/a", "nvidia/b", "cloud/one", "cloud/two"], {})):
            report = engine.scan(ttft=1, attempts=1, stagger=0, key="k", config_path=self.config,
                                 configured_dir=self.state_dir, survey=True, survey_ttft=1)
        self.assertEqual(len(report["models"]), 4)          # everything was measured
        cached = render.render_cached(report)
        self.assertNotIn("cloud/one", cached)               # strangers stay out of the chat view
        self.assertIn(" nvidia/a", cached.replace("—", " "))
        board = render.render_catalog(report)
        self.assertIn("✔️ 4 answering", board)               # but the board shows all four

    def test_a_chat_scan_does_not_erase_the_board(self):
        """The bug behind "it still says five models": /nim used to overwrite the survey's 85 verdicts."""
        def fake_probe(_base, model, _key, _ttft, attempts, _stagger):
            return {"model": model, "status": "ok", "ok_attempts": 1, "attempts": attempts,
                    "ttft_s": 0.4, "trials": ["ok:0.4"], "detail": "",
                    "checked_at": engine._now_iso()}

        catalog = ["nvidia/a", "nvidia/b", "cloud/one", "cloud/two"]
        with mock.patch.object(engine, "probe", side_effect=fake_probe), \
                mock.patch.object(engine, "catalog_entries", return_value=(list(catalog), {})):
            engine.scan(ttft=1, attempts=1, stagger=0, key="k", config_path=self.config,
                        configured_dir=self.state_dir, survey=True, survey_ttft=1)
            after_survey = engine.load_state(self.state_dir)
            engine.scan(ttft=1, attempts=1, stagger=0, key="k", config_path=self.config,
                        configured_dir=self.state_dir)
            after_chat = engine.load_state(self.state_dir)
        self.assertEqual(len(after_survey["models"]), 4)
        self.assertEqual(len(after_chat["models"]), 4)          # board intact
        self.assertEqual(after_chat["survey"]["measured"], 4)   # survey info carried forward
        self.assertEqual(len(render.render_catalog(after_chat).splitlines()), 
                         len(render.render_catalog(after_survey).splitlines()))

    def test_a_survey_never_alerts_about_models_you_do_not_point_at(self):
        def fake_probe(_base, model, _key, _ttft, attempts, _stagger):
            return {"model": model, "status": "http_404", "ok_attempts": 0, "attempts": attempts,
                    "ttft_s": None, "trials": ["http_404"], "detail": "",
                    "checked_at": engine._now_iso()}

        with mock.patch.object(engine, "probe", side_effect=fake_probe), \
                mock.patch.object(engine, "catalog_entries", return_value=(["cloud/one"], {})):
            first = engine.scan(ttft=1, attempts=1, stagger=0, key="k", config_path=self.config,
                                configured_dir=self.state_dir, survey=True, survey_ttft=1)
            second = engine.scan(ttft=1, attempts=1, stagger=0, key="k", config_path=self.config,
                                 configured_dir=self.state_dir, survey=True, survey_ttft=1)
        # the two models in config may speak (they are yours); the stranger in the catalog may not
        self.assertFalse([c for c in first["changes"] if "cloud/one" in c])
        self.assertFalse([c for c in second["changes"] if "cloud/one" in c])
        self.assertEqual(second["models"]["cloud/one"]["status"], "http_404")  # still recorded

    def test_recheck_can_be_disabled(self):
        self._caps(recheck=0)
        self.assertEqual(self._caps(recheck=0), {"nvidia/a": 25, "nvidia/b": 25})

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
