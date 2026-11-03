# hermes-nim

NVIDIA NIM model health, inside Hermes Agent — because NVIDIA will not tell you which of its
models still answer.

```
/nim
NVIDIA NIM · 6 min ago · your 11 models · 17/85 listed answer

🥇 nemotron-3-ultra-550b-a55b — 🏛1445 family · ⚡0.76s · ✍️121.5t/s
🥈 mistral-large-4-1t-a52b — 🏛1430 family · ⚡0.59s · ✍️100.4t/s
🥉 nemotron-3.5-super-vl-120b-a12b — 🏛1405 family · ⚡0.5s · ✍️234.6t/s
· muse-glimmer-30b — 🏛≈1389 family · ⚡1.56s · ✍️7.5t/s · shortlist
· nemotron-3.5-lightning-30b-a3b — 🏛1331 family · ⚡0.45s · ✍️199.5t/s · free
· nemotron-3-nano-omni-30b-a3b-reasoning — ⚡0.26s · ✍️178.9t/s · shortlist

⚠️ minimax-m3.1-flash — 🏛1431 family · rate limited · shortlist

🔴 gpt-oss-20b — 🏛1287 · stalled · fallback 4
🔴 step-5-preview — 🏛1443 · stalled · shortlist

→ best for chat: nvidia/nemotron-3-ultra-550b-a55b — 🏛1445 · ⚡0.76s · 1/1 · ✍️121.5t/s
board: 17 of 85 measured models answer · 85 listed · surveyed 6 min ago
```

Real output, one account: two models in the same fallback chain are dead, one is throttled, and the
three medals are positions in the displayed order — not a verdict about the models' quality beyond the
numbers printed next to them. `≈` marks a score inside the previous row's confidence interval.

**The glyphs are the labels:**

| Glyph | Means | Source |
|---|---|---|
| 🥇🥈🥉 | positions 1-3 of the displayed order | this plugin |
| ≈ | statistically tied with the row above | the source's own interval |
| 🧠 | Hermes Index, 0-100 agentic | `portal.nousresearch.com/bench`, fetched and dated |
| 🏛 | LMArena Elo | LMArena, fetched and dated |
| ⚡ | measured time to first byte | this plugin, just now |
| ✍️ | measured tokens per second | this plugin, just now |
| 🆕 | arrived in the catalog within 7 days | catalog history, this plugin |
| ⚠️ | degraded (flaky, throttled, rate limited) | this plugin |
| 🔴 | dead (stalled, retired, not provisioned) | this plugin |
| ⌛️ | age of the number in the header | this plugin |
| 🗒 | catalog size | `/v1/models` |
| 🟢 | chat models answering · 🛠 utility models | this plugin |

**One scale per column, never averaged.** 🧠 and 🏛 measure different things, so `order_by` decides
which one leads, and that table alone sets the order, the printed score and the medals; models it does
not rate sort below it. `🏛1445 family` means the score belongs to the model family, not to that exact
variant.

- **`/nim`** — a live survey of every listed model (~30 s). No agent turn, no tokens.
- **`/nim cached`** — the last scan, instantly, no network.
- **`/nim probe [model…]`** — your config models plus the shortlist, or exactly the ones you name.
- **`hermes nim`** — the same engine from a shell (`--json` for machines).
- **`nim/background.py`** — keeps the state fresh on its own, no cron required (see
  [Unattended watching](#unattended-watching)).

### Realtime by default

`/nim` measures. A cached verdict still asks the reader to decide whether "22 min ago" means
anything, and the entire reason this plugin exists is that published metadata lies — so the default
path goes to the wire.

Two things make that bearable:

- **The reply is posted before the wait.** The handler posts `⏳ NVIDIA NIM — measuring 7 models live
  (2× each, up to 25s per attempt)…` and then *edits that same message* with the report, so there is
  no 40-second silence and no leftover progress line.
- **A scan younger than `live_grace_seconds` (20 s) is reused.** Double-tapping `/nim` does not fire
  two full probes at the same model.
- **One attempt, every model at once.** Two attempts in two waves took 61 s when something was
  wedged; now all seven probe concurrently with a single sample, so the floor is one stall cap
  (~26 s worst case, typically 1–14 s). `/nim probe` still takes two samples when you want the
  authoritative verdict.

The wait is `ttft_seconds` for a wedged model — that cap is not padding, it is how long you must
listen before "no answer" is a fact. Lower it and slow-but-alive models get reported as dead.

`/nim cached` is the instant view, `live_default: false` puts the old behaviour back, and
`NIM_LIVE=0` does it for one process. `hermes nim` follows the same default; `hermes nim cached` is
the fast door for scripts.

### Titles, bold, and the terminal

The title is `**NVIDIA NIM**` — bold, no glyph, because the header is the one line that should not
compete with the data for attention. Chat is the primary surface and Telegram renders
MarkdownV2, so the canonical render carries the `**`.

`hermes nim` prints through `render.plain()`, which strips the emphasis: a terminal must not show
literal asterisks around a title, and teaching a second renderer every line would let the two doors
drift. The glyphs stay in both — they are content, not decoration.

### Why it looks like that

- **One model, one line.** A phone gets ~40 characters before it wraps; the old lines were 95-110 and
  turned seven models into twenty lines of broken text.
- **The org prefix is dropped in lists** (`nvidia/nemotron-…` → `nemotron-…`) because it is noise in
  every command that is not a config edit. Views where you *act* — `/nim dead`, `/nim cfg`, the
  recommended footer, `/nim json` — keep the full id, so it is always copy-pasteable from somewhere.
- **No repeated size.** `nemotron-3.5-lightning-30b-a3b` already says 30B; the tag only appears when
  the id does not say it.
- **A blank line instead of a header** between healthy, flaky and dead models.
- **`🥇🥈🥉` mark positions, one per position, and the footer names the chat pick once with its
  evidence.** A tie is `≈` on the score, never a duplicated medal.
- Roles are shortened (`free route` → `free`) but not abbreviated to codes nobody can read.

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
| `/nim` | **live scan — measures every model now** | ~15–40 s |
| `/nim cached` | the last scan, with its age, no network | instant |
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

## Ordering — and what `→ best for chat` means

Two fetched tables rate models, on two different scales, and **`order_by` decides which one leads**:

- **🧠 Hermes Index** (`portal.nousresearch.com/bench`) — 0-100, agentic, ~18 rows.
- **🏛 LMArena** — Elo from human preference, the widest coverage of a NIM account by far.

Neither is a superset, so they are never averaged. The leading table — `arena` by default — sets the
list order, the score printed on each row, and the medals; models it does not rate sort below it and
cannot take 🥇. Every score keeps the source's own confidence interval, which is what `≈` prints: two
rows whose intervals overlap are one number twice, and the medal still goes to one of them because the
reader asked for one pick per position.

`→ best for chat` is a narrower claim than 🥇, and it is normal for them to disagree: the pick must be a
chat model (not a guard or an embedder), answer within `recommend_max_ttft_s` (5 s), be graded `ok`, and
a scored model must also write at `recommend_min_tps` (25 t/s) to keep the top tier. Ranking alone is
not a recommendation — the smartest model on the board is useless in a chat loop if it takes 40 s to
start talking — so speed is a gate, not a tie-break. Exactly one model is named, or none is.

That distinction is the point of the two: ranking answers "what is strongest", the footer pick answers
"what should I use right now". They disagree whenever the strongest model is slow.

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
| `recheck_ttft_seconds` | `8` | a model already marked dead only has to emit a byte (`0` = use `ttft_seconds`) |
| `stagger_seconds` | `1.5` | pause between attempts |
| `state_dir` | `~/hermes-outputs/nim-probe` | shared history; set the same path across profiles |
| `shortlist` | empty | extra models to watch that are *not* in your config |
| `live_default` | `true` | `/nim` measures; `cached` is a separate verb |
| `live_attempts` | `1` | interactive `/nim`; `/nim probe` uses `attempts` |
| `live_grace_seconds` | `20` | a double-tap reuses a scan younger than this |
| `survey_catalog` | `true` | `/nim` surveys every listed model |
| `survey_workers` | `20` | 20 measured 12 s for 85 models; 28 gained nothing and hit 429 |
| `survey_ttft_seconds` | `10` | the survey's own cap, so 9 black-holed models cost 10 s, not 225 s |
| `order_by` | `arena` | which table leads: `arena` 🏛 · `index` 🧠 · `speed` ✍️ |
| `recommend_max_ttft_s` | `5` | the footer pick must answer this fast to stay eligible |
| `recommend_min_tps` | `25` | a scored model must write this quickly to keep the top tier |
| `scores_refresh` / `arena_scores` | `true` | refetch the indexes when stale; fill gaps from LMArena |
| `background_config_scan_min` | `10` | how often the scanner re-probes your config models (`0` = off) |
| `background_survey_min` | `60` | how often it surveys the whole catalog (`0` = off) |
| `language` | `auto` | `auto` follows `display.language`; `en` / `es` pin it |

## Unattended watching

The plugin watches on its own. `nim/background.py` starts a daemon thread with its own asyncio loop when
the plugin loads — no cron job, no gateway loop, nothing to schedule:

- **config scan**, every `background_config_scan_min` (10) — the models your config points at plus the
  shortlist, which is the data the fallback chain depends on;
- **catalog survey**, every `background_survey_min` (60) — every listed model, which keeps `/nim
  catalog`, `/nim added`, `/nim dead` and the medals current.

Both write the same `state.json` `/nim cached` reads, so the chat door and the background never
disagree. Every sweep logs a start and a done line with counts; a scanner that stops working is
therefore visible in `~/.hermes/logs/gateway.log` instead of looking like a quiet one.

`scripts/nim-watch.py` still exists for people who want a message rather than a fresh state file: it is
a `no_agent` cron script whose **stdout is the message and empty stdout sends nothing**, so a healthy
tick is silent. A model is only announced as dead once two consecutive runs agree (and only once, until
it recovers) — without that hysteresis a flapping provider produces alarm spam every tick. Do not pass
`--send` when the cron engine already delivers stdout, or every alert arrives twice.

## Honest limits

- **No usage or quota data.** The free tier exposes no per-account usage endpoint; the only quota
  signal is a `429`. `/nim` reports throttling, not token spend. Real accounting needs
  instrumentation at the provider layer, not at the model-health layer.
- **The catalog survey is a single sample at a 10 s cap.** 85 models × 25 s would be an hour, so the
  survey asks each id for one byte within 10 s; `/nim probe` is the door that spends the full cap and
  two attempts on the models you actually route to.
- **A quiet report is not a guarantee.** A model that answered during the scan can wedge a minute
  later, which is why the scanner keeps re-measuring instead of trusting one board.
- **A leaderboard is a prior, not a verdict.** Both sources are fetched, dated and cached — never
  typed — and both rate a minority of what NIM serves, so an unrated model is printed as unrated rather
  than given a proxy score. A big id is not evidence: a 550B model can be beaten by a 428B one, and
  measured speed is frequently the opposite of the ranking.

## Localization

Every user-facing string lives in `locales/`, not in the Python. Shipped: **en** and **es** (Mexican
register — a colleague, not a manual).

```
/nim cached          →  NVIDIA NIM · hace 6 min · tus 11 modelos · 17/85 listados responden

                        🥇 nemotron-3-ultra-550b-a55b — 🏛1445 familia · ⚡0.76s · ✍️121.5t/s
                        🥈 mistral-large-4-1t-a52b — 🏛1430 familia · ⚡0.59s · ✍️100.4t/s
                        · muse-glimmer-30b — 🏛≈1389 familia · ⚡1.56s · ✍️7.5t/s · candidato

                        → mejor para chat: nvidia/nemotron-3-ultra-550b-a55b — 🏛1445 · ⚡0.76s · 1/1
```

`/nim help` and the settings block are written in rich markdown (headings, bullets, a table for the
glyphs), so Telegram renders them client-side through Bot API 10.1's `sendRichMessage` — which means
copying the message keeps the formatting. The tabular reports stay on MarkdownV2 on purpose: a client
renderer would collapse the column alignment.

Language resolution, first hit wins: an explicit `lang=` → `NIM_LANGUAGE` → `HERMES_LANGUAGE` →
`plugins.entries.hermes-nim.settings.language` → `display.language` → `en`. So set it once in Hermes
config and `/nim` follows the rest of the agent.

Adding a language is one file: copy `locales/en.yaml` to `locales/pt.yaml`, translate the values, keep
the `{placeholders}` and the glyphs, add the id to `provides_locales` in `plugin.yaml`. A test fails
if any shipped language is missing a key that English has, so a half-translated pack cannot ship
silently.

Two deliberate choices worth knowing:

- **Key resolution is our own, not `agent.i18n.t`.** The host layer needs a repo-root `hermes_yaml`
  module on `sys.path` (it warns and returns bare keys when a plugin runs outside the install dir) and
  resolves language from a config cache we cannot pin. Ours is deterministic, silent, and identical in
  the gateway, the CLI and the cron script.
- **A missing key degrades to English, then to the key itself** — never to a traceback. A watchdog that
  dies because a translation file has a typo is worse than an English sentence.

Per-line overrides: drop `$HERMES_HOME/locales/es.yaml` with just the keys you want to change, e.g.

```yaml
rec:
  line: "→ mejor para chat: {model} — {evidence}"
```

Your overlay outranks the shipped file, so a rewording costs one line and no fork.

## What the star means

The ⭐️ is "point Hermes at this one", which is a different question from "which model ranks first":

1. **A published score and a real writing speed** — quality, confirmed usable.
2. **A published score that crawls** — still the smartest thing available, just leaden.
3. **No score, but a measured writing speed** — the fastest evidence available.
4. **No score, no speed** — first byte only.

Gates: it must be a chat model (a safety classifier answering in 0.05 s is not one), it must have
answered every attempt, and its first byte must clear `recommend_max_ttft_s` (5 s). Parameter count is
deliberately **not** part of this ordering — "550B in the id" is a guess about quality, and it used to
put the star on the biggest unscored model rather than the one that was demonstrably pleasant.

## What the answer tells you

The probe reads the reply, not just the status code. Measured on this account:

```
openai/gpt-oss-20b           ⚡0.34s  ✍️36.9t/s   32 tokens   req e5ba9f4f
minimaxai/minimax-m3.1-flash ⚡1.46s  ✍️24.3t/s   32 tokens   req b49c3b3b
```

- **✍️ tokens/second** — generation speed, from `usage.completion_tokens` over the time after the
  first byte. Only shown when ≥4 tokens came back: below that the rate is noise.
- **🏢 vendor split** — `owned_by` from `/v1/models`, counted over the models that answered. Turns
  "13 models are down" into "one lab is down".
- **req** — `nvcf-reqid`, the only handle NVIDIA support can search on. Shown on dead rows.

Two things that are returned but useless here, so they are captured and not advertised:
`service_tier` and `system_fingerprint` are `null` on this gateway, and streaming chunks omit both
entirely. If NVIDIA starts populating them, the footer picks them up with no code change.

## The board: who is actually online

NVIDIA lists 85 models for this account. 14 answer. The other 71 are 404-unprovisioned, 500ing, or
black-holing — and `/v1/models` calls all of them available. That gap is the reason this plugin exists,
so it is measured, not read:

```
NVIDIA NIM catalog — 85 models listed · surveyed 7 min ago
✔️ 17 answering: nemotron-3-ultra-550b-a55b, mistral-large-4-1t-a52b, muse-glimmer-30b, …
🔴 68 not useable:
  • nemotron-parse — http_400
  • yi-large — not provisioned
  • fuyu-8b — not provisioned
  … 63 more (all of it in /nim json)
```

`/nim survey` measures all of them now — **~12 s** for 85 models at 20 workers, one attempt each,
10 s cap (a sweep at the 25 s fallback cap would take an hour). The background scanner runs the same
survey every `background_survey_min` (60), so the board is never older than an hour.

Three rules keep that from becoming noise:

- **`/nim` is the survey.** Every tap measures all 85 (17 s) and prints the whole board: chat models
  ranked, utility models by name, offline by reason. `/nim probe` is the fast door for the models your
  config uses; `/nim cached` reprints the last survey without touching the wire.
- **Only models you point at may alert.** Confirming that `yi-large` is still 404, every six hours,
  forever, is not monitoring.
- **The board is a survey, not a verdict for use.** One attempt, ten-second cap: enough to say
  "answers" or "does not", not enough to rank. Use `/nim speed` for latency you can reason about.

You do not maintain a list of names. Adding a model to your config makes it watched; the board tells
you what else exists; `/nim survey` refreshes on demand.

## Rechecking a dead model costs less

A verdict is not a subscription to 25 s of silence. Once a model is confirmed dead, the next sweep
asks it a cheaper question: did *any* byte come back within `recheck_ttft_seconds` (default 8)?

Measured on a model that black-holes (same endpoint, same state, one attempt each):

```
recheck_ttft=8   32.8 s wall   trials ['stall:8.12']    # the dead one gave up at 8 s
recheck_ttft=0   44.6 s wall   trials ['stall:20.14']   # full cap, to learn the same thing
```

The shorter cap is not a softer claim about the model: a NIM endpoint that needs more than ~8 s to
emit its first byte is already unusable in a fallback chain, where the recommendation gate is 5 s. A
slow recovery that exceeds the cap is not a recovery worth reporting — and a model that *does*
answer gets the full `ttft_seconds` on its next run, so the shortcut never grades a live model.

Set it to `0` to disable, then every model costs the full cap again.

## Versioning

Four numbers, and the last one is where most releases go:

```
0   .  0   .  7   .  1
│      │      │      └─ cosmetic or tiny fix — formatting, wording, a glyph
│      │      └──────── behaviour change: a verb, a default, a new verdict
│      └─────────────── capability area (probe engine / plugin surface / watchdogs)
└────────────────────── breaking change to the output contract — 1.x when it happens
```

So a day of formatting work moves `0.0.7.1 → 0.0.7.4`, not `1.7.1 → 1.10.0`. The version exists to
answer one question — "is what I installed the thing I am reading about?" — and nothing else, so it
should not look like a product launch every time a line gets shorter.

`plugins update` compares these with PEP 440 ordering, which handles four segments correctly
(`0.0.7.2 > 0.0.7.1`). Renumbering *down* from `1.7.1` to `0.0.7.1` cannot be pulled by an update, so
that one move needs `hermes plugins install brewmaister/hermes-nim --enable --force` — a reinstall,
once, which the changelog entry for this version says plainly.

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
