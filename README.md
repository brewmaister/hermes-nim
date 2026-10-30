# hermes-nim

NVIDIA NIM model health, inside Hermes Agent — because NVIDIA will not tell you which of its
models still answer.

```
/nim
🧠 NIM — last scan 20 min ago · 7 models

✅ minimaxai/minimax-m3.1-flash — AA 29 family · ok 0.57s (2/2) · shortlist (recommended)
✅ nvidia/nemotron-3-ultra-550b-a55b — AA 23 · ok 0.53s (2/2) · fallback 2, moa reference
✅ nvidia/nemotron-3.5-lightning-30b-a3b — 30B · ok 0.29s (2/2) · free route, fallback 1
✅ meta/muse-glimmer-30b — 30B · ok 3.28s (2/2) · shortlist
✅ openai/gpt-oss-20b — 20B · ok 0.34s (2/2) · fallback 4
⚠️ stepfun-ai/step-5-preview — FLAKY — only 1/2 answered (worst stall 25s) · shortlist
❌ deepseek-ai/deepseek-v4.1-flash — STALL — zero bytes on all 2 attempts · fallback 3

recommended: minimaxai/minimax-m3.1-flash — AA 29, 1000k ctx, 2/2 answered, 0.57s
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
| `/nim speed [n]` | timed scan: first byte + full answer, median of n attempts (default 3) | ~15–40 s |
| `/nim catalog` | what `/v1/models` lists, what dropped out | instant |
| `/nim dead` | everything currently unusable | instant |
| `/nim added [days]` | models first seen recently (default 7) | instant |
| `/nim cfg` | which NIM models your config references | instant, no network |
| `/nim json` | the last scan as JSON | instant |
| `/nim version` | the build the running process imported | instant |

`/nim probe` is the slow one on purpose: a model that never answers costs two 25 s stalls before it
can honestly be called dead. Everything else renders cached state.

## Ordering — and what "(recommended)" means

Models that answered are listed first, strongest first, on evidence in descending order of trust:

1. **A published benchmark index** (`nim/benchmarks.py`) — Artificial Analysis Intelligence Index,
   with the source URL and the date it was read stored beside it. Minimax-M3 scores **29**,
   Nemotron-3-Ultra **23**. External evidence, labelled as external: the line says `AA 29`, and a
   variant with no published score of its own says `AA 29 family` so nobody reads it as a
   measurement of that exact model.
2. **Otherwise the parameter count in the model id** — `nemotron-3.5-lightning-30b-a3b` → 30B. A
   weak prior, but it is the vendor's own number rather than a guess, and it beats ranking by
   nothing. Models with neither an index nor a stated size sort last and show no number at all.
3. **Then measured behaviour** — attempts-answered ratio, then time to first byte.

**(recommended)** is a separate, narrower claim: the highest-ranked model that answered *every*
attempt **and** emits a first byte within 5 s. Ranking alone is not a recommendation — the smartest
model on the board is useless in a chat loop if it takes 40 s to start talking — so speed is a gate,
not a tie-break. Exactly one model carries the tag, or none does.

That distinction is the point of the two: ranking answers "what is strongest", the tag answers "what
should I use right now". They disagree whenever the strongest model is slow.

## When the key is the problem

A revoked `NVIDIA_API_KEY` looks *exactly* like a wiped model platform: every probe returns 401. So
`/nim` refuses to write that into the model history. When every probed model is rejected by the same
response, the scan stops and reports the credential instead:

```
🧠 NIM — 7 models known

🔑 NVIDIA key rejected — HTTP 401: Authorization failed
   Models were NOT probed: this is the credential, not the models.
   Fix NVIDIA_API_KEY in ~/.hermes/.env, then /nim probe.
   Below is the last good scan (18 min ago), unchanged.
```

The previous verdicts stay untouched, so fixing the key does not produce a fake "RECOV" for every
model. The condition is announced when it appears and then at most once every 24 h while it lasts.
`/nim speed` answers `cannot measure` rather than timing a wall.

Detection is unanimity across the model probes, not `/v1/models`: NVIDIA serves that catalog to any
key, so a revoked key still lists 85 models and only `/chat/completions` says 401.

## Timing models: /nim speed

`/nim speed [n]` (default 3 attempts) reports two different numbers per model, both medians:

- **first byte** — how long before it starts talking; this is what makes a model feel fast
- **to finish** — the whole 32-token answer, which tracks tokens/s

```
1. nvidia/nemotron-3.5-lightning-30b-a3b — 0.28s (0.28 / 1.04 / 0.28) · 1.34s to finish
2. openai/gpt-oss-20b — 0.42s (0.49 / 0.29 / 0.42) · 1.04s to finish
7. stepfun-ai/step-5-preview — 16.22s (16.22 / 0.39 / 22.16) · 16.87s to finish
```

Median, not mean, on purpose: one 20 s stall is a real event but it is not the model's speed, and a
mean would report it as though it were. Models that never emitted a byte are listed separately, not
quietly dropped.

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
- **The benchmark table is small and hand-maintained.** Two families have published scores; every
  other model falls back on its id's parameter count, which is a much weaker signal — a 550B model
  can be beaten by a 428B one, as MiniMax-M3 beats Nemotron-3-Ultra on the index while carrying
  fewer parameters. Treat the ordering as a prior, not a verdict.

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
