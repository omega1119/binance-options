# Crypto Options on Binance — Pricing, Greeks, IV (Notebook + Code)

Show understanding of options with **working examples**: Black–Scholes, **Binomial (CRR)**, and **Monte Carlo**, plus **implied volatility** and **Greeks** — using **Binance Options** data where available.

## What’s inside

- `Binance_Options_Pricing_Tutorial.ipynb` — a self-contained walkthrough that:
  - Pulls **live** Binance Options data (mark prices, IVs, Greeks) or uses a bundled **sample**.
  - Reprices quotes with **Black–Scholes**, rebuilds **IV**, and plots an **IV smile**.
  - Cross-checks **Binomial** and **Monte Carlo** prices.
  - Validates **Greeks** against Binance’s feed.
- `src/option_math.py` — clean implementations of:
  - Black–Scholes pricing & Greeks (with continuous yield)
  - Implied volatility (bisection)
  - Cox–Ross–Rubinstein (binomial tree)
  - Risk‑neutral Monte Carlo
- `src/binance_adapter.py` — tiny helper to hit the Binance Options REST endpoints.
- `data/sample_mark_data.json` — realistic sample so the notebook runs offline.

## Quickstart

```bash
git clone https://github.com/omega1119/binance-options
cd binance-options
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab  # or jupyter notebook / VS Code
```

Open **`Binance_Options_Pricing_Tutorial.ipynb`**, run all cells.  
- By default it uses **sample data** (no network).  
- To fetch **live Binance** data, set in the first code cell:

```python
USE_LIVE = True
UNDERLYING = "BTCUSDT"  # or "ETHUSDT"
```

## Binance endpoints used

- **Mark (price & greeks):** `GET /eapi/v1/mark`  
- **Exchange info (symbols):** `GET /eapi/v1/exchangeInfo`  
- **Underlying index price:** `GET /eapi/v1/index`

## Requirements

```
python>=3.10
numpy
pandas
matplotlib
requests
jupyter
```

Installed via:

```bash
pip install -r requirements.txt
```