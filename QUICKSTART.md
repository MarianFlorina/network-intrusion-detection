# ⚡ Quickstart — three ways to run

> **Data policy:** the dashboard, predictions and monitoring windows are
> **real-traffic-only** (source `capture`, from your NIC via Mode 2). Synthetic
> flows exist solely as the labelled **bootstrap training set** and (after
> adaptation) as class anchors — they never appear in the live views.

## Mode 1 — Local demo (5 minutes, no admin needed)

Bootstraps the model on **synthetic** traffic (labelled training data), then
switch to Mode 2 to feed it your real network. Works on any laptop with
Python 3.10–3.13.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt        # Linux/macOS

.venv\Scripts\python -m scripts.run_local          # full MLOps lifecycle demo
.venv\Scripts\python -m streamlit run dashboard/app.py --server.port 8501
```

Then open http://localhost:8501 — the Network Security Monitor, with real DB
data (synthetic traffic). Optional extras:

```bash
.venv\Scripts\python -m pytest tests -q            # 34 tests
.venv\Scripts\mlflow ui --backend-store-uri file:./mlruns   # experiment UI
```

## Mode 2 — Real traffic capture (the default runtime)

Scores YOUR laptop's actual network traffic — the only data shown on the
dashboard and consumed by monitoring/adaptation.

Windows one-time setup:
1. Install Npcap → https://npcap.com (tick **"Install Npcap in WinPcap API-compatible Mode"**)
2. Open the terminal **as Administrator** (Start → type "terminal" → right-click → Run as administrator → `cd` to this folder)

```bash
.venv\Scripts\python -m scripts.capture --check              # must end: Environment OK
.venv\Scripts\python -m scripts.capture --list-interfaces    # find Wi-Fi / Ethernet
.venv\Scripts\python -m scripts.capture --window 60 --iface "Wi-Fi"     # one window
.venv\Scripts\python -m scripts.capture --continuous --interval 60 --iface "Wi-Fi"
```

Watch the dashboard (Traffic source → `capture`). Ctrl+C stops within ~1 s.
Linux/macOS: libpcap is usually preinstalled; run the same commands with `sudo`.

## Mode 3 — Full Docker stack (all five UIs)

Requires **Docker Desktop**. This is the architecture-diagram mode: real
PostgreSQL, MLflow server, Airflow UI, FastAPI, dashboard.

```bash
copy .env.example .env          # macOS/Linux: cp .env.example .env
docker compose up --build -d
```

| Service | URL | Login |
|---|---|---|
| Airflow UI | http://localhost:8080 | admin / admin |
| MLflow UI | http://localhost:5000 | — |
| FastAPI docs | http://localhost:8000/docs | — |
| Streamlit | http://localhost:8501 | — |

In Airflow, unpause both DAGs, trigger `network_ids_training` once, then
`network_ids_monitoring`. Drift demo trigger:

```bash
docker compose exec airflow-scheduler airflow dags trigger network_ids_monitoring ^
  -c '{"day":"attack-wave","drift_profile":{"src_bytes_scale":2.8,"packet_rate_shift":1.8,"attack_rate":0.12}}'
```

## Mode 4 — Adapt the model to your real network (after Mode 2)

```bash
.venv\Scripts\python -m scripts.adapt --status   # captured flows vs needed (3000)
.venv\Scripts\python -m scripts.adapt --run      # gated self-training cycle
```

# Does it work perfectly on everyone's laptop?

**The pipeline: yes, by design.** Mode 1 is pure Python + SQLite + a file-based
MLflow store — no admin rights, no Docker, no special drivers, any OS. If
`pip install -r requirements.txt` succeeds, the demo runs. Python 3.13 needs
nothing extra; the pinned versions have wheels for 3.10–3.13.

**Real capture (Mode 2): mostly, with per-OS caveats:**

| Concern | Reality |
|---|---|
| Windows | Needs Npcap + Administrator. Without them, the tool *tells you* exactly what's missing instead of crashing. |
| Linux/macOS | libpcap usually present; `sudo` needed. Wi-Fi monitors see only your own traffic (same as Windows). |
| Wi-Fi vs Ethernet | Works on both — the tool auto-detects interfaces (`--list-interfaces`). |
| Antivirus | Some (e.g. certain EDR/AV suites) block raw packet capture; if `--check` passes but `sniff` fails, that's the usual suspect. |
| Laptop sleep / Wi-Fi power save | Produces empty windows (`no flows captured`) — expected, not a bug. |
| VPN | Captures your VPN's virtual adapter traffic; use `--list-interfaces` to pick it. |

**What "perfectly" does NOT mean — read this before demoing:**

1. **Flagged attacks on real traffic are expected false positives** (~5–13%).
   The model was trained on synthetic data; your real traffic statistically
   resembles those signatures. The adaptation cycle (Mode 4) exists to fix this.
2. **Zero detected attacks ≠ zero real attacks.** Absence of ground truth means
   the anomaly metrics are only meaningful relative to the training data.
3. **Airflow/MLflow UIs (Mode 3) require Docker installed** — on a fresh laptop
   that's a ~500 MB install. Mode 1 needs none of it.
4. **Numbers won't match another laptop's** — traffic is personal (different
   apps, background services). The *pipeline behavior* is identical; the *data*
   is yours.
