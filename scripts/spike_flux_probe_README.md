# spike_flux_probe.py

Feeds a WAV to the real Deepgram Flux service at 20 ms per chunk and writes every frame and
event it sees to JSONL, so the FluxStub of the 8a spike can be checked against real timings.

```bash
# From the repo root, with the api venv. Clips must be 16 kHz mono PCM16.
DEEPGRAM_API_KEY=... PYTHONUTF8=1 python scripts/spike_flux_probe.py clip-a.wav out/clip-a-0.5.jsonl --eager 0.5
```

Options: `--eager` (eager_eot_threshold, default 0.5) and `--eot-timeout-ms` (default 3000, the
production value; the silence tail sent after the clip is this plus 500 ms). Model, eot_threshold
and the Spanish language hint are fixed to the production values.

Exit codes: `0` connected and at least one StartOfTurn - `1` bad input or crash - `2` connection
failed - `3` connected but no StartOfTurn - `4` Flux's watchdog fired, so the pacing broke and the
timings are worthless - `5` the pipeline did not drain. Never 0 without a connection and a
StartOfTurn. Note that argparse also exits 2 on a usage error; a real connection failure is the
one that leaves a JSONL behind, carrying `connection_error` or `CONNECT_FAILED`.

The spike runs 6 clips against `--eager 0.5` and `--eager 0.3` (12 JSONL). Neither the WAV clips
nor the JSONL belong in this repo: they live in the docs repo under
`docs/2026-09-09-8a-hybrid-turn-viability/`.
