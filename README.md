# hermes-nim

NVIDIA NIM model health, inside Hermes Agent — because NVIDIA will not tell you which of its
models still answer.

```
/nim
🧠 NIM — last scan 20 min ago · 7 models

✅ nvidia/nemotron-3-ultra-550b-a55b — 550B · ok 0.53s (2/2) · fallback 2, moa reference ← best
✅ nvidia/nemotron-3.5-lightning-30b-a3b — 30B · ok 0.29s (2/2) · free route, fallback 1
✅ meta/muse-glimmer-30b — 30B · ok 3.28s (2/2) · shortlist
✅ openai/gpt-oss-20b — 20B · ok 0.34s (2/2) · fallback 4
✅ minimaxai/minimax-m3.1-flash — ok 0.57s (2/2) · shortlist
⚠️ stepfun-ai/step-5-preview — FLAKY — only 1/2 answered (worst stall 25s) · shortlist
❌ deepseek-ai/deepseek-v4.1-flash — STALL — zero bytes on all 2 attempts · fallback 3

best right now: nvidia/nemotron-3-ultra-550b-a55b — 550B, 2/2 answered, 0.53s
catalog: 85 listed · 85 tracked
```

- **`/nim`** — the last scan, instantly. No agent turn, no tokens.
- **`/nim probe`** — live scan of the NIM models your config uses plus a shortlist of candidates.
- **`hermes nim`** — the same engine from a shell, for scripts and cron (`--json` for machines).
- **`scripts/nim-watch.py`** — the cron door: prints nothing unless something changed.

## Why this exists

NVIDIA's free NIM tier rots silently, and every published signal is wrong:

| Signal | What it actually tells you |
|---|---|
| `build.nvidia.com` model card | "Free Endpoint — Available" for endpoints that return **zero bytes** |
| `GET /v1/models` | lists ids this account gets `404 not provisioned` on |
| `status.ngc.nvidia.com` | infrastructure only — stayed green through every death below |
| Error body | the only honest source: `410 … reached its end of life on 2026-10-28T08:00:00Z` |

Measured on one day: `z-ai/glm-5.3` retired with no announcement, `z-ai/glm-5.3-flash` went from
answering to zero bytes while still listed, `openai/gpt-oss-20b` was 6/6 healthy and then 6/6
stalled fifteen minutes later, `deepseek-ai/deepseek-v4.1-flash` alternated stall/ok in a strict
pattern (one wedged replica behind the gateway), and a brand-new `mistral-large-4` turned
unprovisioned twelve hours after it appeared.

So the only trustworthy measurement is a probe, and **one probe is a coin flip** — which is why
every model gets several attempts and a model that answers only some of the time is reported as
`flaky` rather than `ok`.

## Install

```bash
hermes plugins install brewmaister/hermes-nim --enable
```

Per profile, not per box: add `-p <profile>` to install and enable.

Then in any chat session: `/nim`. In a shell: `hermes nim`.

Requires a `NVIDIA_API_KEY` (`nvapi-…`) in `~/.hermes/.env`, which is the same key the provider
itself uses — the plugin reads it from there, or from the environment.

### Staying current — two steps, and the second is not optional

```bash
hermes plugins update hermes-nim
scripts/reload-gateway.sh
```

A plugin's slash command is a handler the gateway imported at startup. Moving the clone on disk does
not re-import it, so `/nim` keeps serving the old code while `hermes nim` (a fresh process) serves
the new one — the staleness is invisible from the shell. `scripts/reload-gateway.sh` reloads in
place through the control socket; it exits non-zero when no gateway answered, so a reload that never
happened can never read as success.

## Commands

| Command | What it does | Cost |
|---|---|---|
| `/nim` | cached scan (the last one, with its age) | instant |
| `/nim probe` | live scan: config models + shortlist | ~30–60 s |
| `/nim catalog` | what `/v1/models` lists, what dropped out | instant |
| `/nim dead` | everything currently unusable | instant |
| `/nim added [days]` | models first seen recently (default 7) | instant |
| `/nim cfg` | which NIM models your config references | instant, no network |
| `/nim json` | the last scan as JSON | instant |
| `/nim version` | the build the running process imported | instant |

`/nim probe` is the slow one on purpose: a model that never answers costs two 25 s stalls before it
can honestly be called dead. Everything else renders cached state.

## Ordering — what "best" means here

`/nim` lists the models that answered first, strongest first, and marks the top pick. The order uses
three facts and nothing else:

1. **Parameter count, parsed from the model id** — `nemotron-3-ultra-550b-a55b` → 550B. NIM exposes
   no benchmarks, but the vendor puts the size in the name, so it is the one quality prior that is
   not an opinion. It is printed on every line so the order is never a mystery.
2. **Reliability** — how many attempts answered. A model that answered 1/2 is never ranked above one
   that answered 2/2.
3. **Latency** — measured time to first byte, last tie-break only.

A model whose id states no size (`minimax-m3.1-flash`) is listed after the known sizes and shown
without one, rather than being handed a made-up number. Fast-but-small never outranks big-and-slow:
`gpt-oss-20b` at 0.34 s sits under `nemotron-ultra` at 0.53 s, deliberately — for real work, 550B of
model beats 0.2 s.

## How a verdict is reached

`stream: true` and time **time-to-first-byte**. A non-streaming call to a wedged model just hangs;
a streaming one tells you it produced nothing. Then several attempts, because the gateway pools
replicas:

| Status | Meaning |
|---|---|
| `ok` | every attempt streamed a token |
| `flaky` | some attempts stalled, at least one answered — the replica-rot case |
| `throttled` | answered, but some attempts got `429` — quota, not rot |
| `stall` | zero bytes on every attempt |
| `eol` | `410`, with the retirement date parsed out of the body |
| `not_provisioned` | `404` — listed in the catalog, not callable by this account |
| `rate_limited` | `429` on every attempt |

Reasoning models (`nemotron-*`, `glm-*`, `gpt-oss`) spend the token budget on `reasoning_content`
before any `content`, so a healthy one looks broken at `max_tokens: 8`. Liveness only needs 32.

## Settings

`config.yaml → plugins.entries.hermes-nim.settings`, or `NIM_*` env vars for cron and one-offs:

| Key | Default | Why |
|---|---|---|
| `api_base` | `https://integrate.api.nvidia.com/v1` | for a proxied or self-hosted gateway |
| `ttft_seconds` | `25` | NIM cold starts run 15–20 s; below 20 misreads them as dead |
| `attempts` | `2` | one call lies |
| `stagger_seconds` | `1.5` | pause between attempts |
| `state_dir` | `~/hermes-outputs/nim-probe` | shared history; set the same path across profiles |
| `shortlist` | three known-good candidates | models to check that are *not* in your config |

## Unattended watching

`scripts/nim-watch.py` is designed for a `no_agent` Hermes cron job: **stdout is the message and
empty stdout sends nothing**, so a healthy tick is silent.

```bash
hermes cron create "every 6h"        # then point the job's script at ~/.hermes/scripts/nim-watch.py
```

Copy `scripts/nim-watch.py` to `~/.hermes/scripts/` and repoint the job there, or run
`hermes nim watch` from your own scheduler. Do not also pass `--send` when the cron engine already
delivers stdout — every alert would arrive twice.

It shares the state file with `/nim`, so the two never disagree, and a model is only announced as
dead once two consecutive runs agree (and only once, until it recovers). Without that hysteresis a
flapping provider produces alarm spam every tick.

## Honest limits

- **No usage or quota data.** The free tier exposes no per-account usage endpoint; the only quota
  signal is a `429`. `/nim` reports throttling, not token spend. Real accounting needs
  instrumentation at the provider layer, not at the model-health layer.
- **The full catalog is not probed.** 85 models × a 25 s stall cap is minutes of waiting. `/nim`
  probes what your config uses plus the shortlist; new arrivals are probed once when they appear.
- **A quiet report is not a guarantee.** A model that answered during the scan can wedge a minute
  later. That is why the watchdog exists as well as the command.

## Development

```bash
~/.hermes/hermes-agent/venv/bin/python -m unittest discover -s tests   # 42 tests, offline
```

The tests drive the engine through a fake transport (no network, no API key): response
classification, retry/throttle semantics, catalog history, hysteresis, rendering and the cron
silence contract. Add a test with any behaviour change — the flapping-provider bugs of the first
version were all invisible without one.

## License

MIT.
