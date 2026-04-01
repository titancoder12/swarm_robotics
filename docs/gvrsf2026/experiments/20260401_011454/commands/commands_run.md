# Commands Run

Primary campaign runner:

```bash
python docs/gvrsf2026/experiments/20260401_011454/commands/run_campaign.py
```

Figure generation with safe local Matplotlib cache:

```bash
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/xdg_cache \
python docs/gvrsf2026/experiments/20260401_011454/commands/plot_from_tables.py
```

Sanity checks used while executing:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python -m py_compile docs/gvrsf2026/experiments/20260401_011454/commands/run_campaign.py
PYTHONPYCACHEPREFIX=/tmp/pycache python -m py_compile docs/gvrsf2026/experiments/20260401_011454/commands/plot_from_tables.py
```
