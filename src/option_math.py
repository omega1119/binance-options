
import math
import random
from dataclasses import dataclass
from typing import Tuple, Literal, Optional, List
import datetime

import numpy as np

OptionType = Literal["C", "P"]

# --- Basic utilities ---

def std_norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def std_norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def year_fraction(start: float, end: float) -> float:
    """Compute ACT/365F year fraction from two POSIX timestamps (seconds)."""
    return max(1e-9, (end - start) / (365.0 * 24.0 * 60.0 * 60.0))

def parse_binance_symbol(symbol: str):
    """
    Parse Binance options symbol like 'BTC-220815-50000-C'.
    Returns (underlying_base, expiry_date(datetime.date), strike(float), opt_type('C'|'P'))
    """
    # Examples: BTC-220815-50000-C, ETH-250118-3500-P
    parts = symbol.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unrecognized symbol format: {symbol}")
    base = parts[0]
    yymmdd = parts[1]
    strike = float(parts[2])
    opt_type = parts[3][0].upper()
    # Parse date as 20YYMMDD (Binance examples are 6 digits YYMMDD)
    yy = int(yymmdd[0:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    year = 2000 + yy if yy < 80 else 1900 + yy  # crude pivot; good for the next decades
    expiry = datetime.date(year, mm, dd)
    return base, expiry, strike, opt_type

# --- Black-Scholes ---

def bs_d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float):
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        # Avoid bad inputs
        return float("-inf"), float("-inf")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_price(S: float, K: float, r: float, q: float, sigma: float, T: float, opt: OptionType) -> float:
    """Black–Scholes price for European options with continuous yield q."""
    d1, d2 = bs_d1_d2(S, K, r, q, sigma, T)
    if opt == "C":
        return math.exp(-q * T) * S * std_norm_cdf(d1) - math.exp(-r * T) * K * std_norm_cdf(d2)
    else:
        return math.exp(-r * T) * K * std_norm_cdf(-d2) - math.exp(-q * T) * S * std_norm_cdf(-d1)

def bs_greeks(S: float, K: float, r: float, q: float, sigma: float, T: float, opt: OptionType):
    """Return greeks: delta, gamma, theta, vega, rho."""
    d1, d2 = bs_d1_d2(S, K, r, q, sigma, T)
    pdf_d1 = std_norm_pdf(d1)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if opt == "C":
        delta = disc_q * std_norm_cdf(d1)
        rho = K * T * disc_r * std_norm_cdf(d2)
        theta = (- (S * disc_q * pdf_d1 * sigma) / (2 * math.sqrt(T))
                 - r * K * disc_r * std_norm_cdf(d2)
                 + q * S * disc_q * std_norm_cdf(d1))
    else:
        delta = -disc_q * std_norm_cdf(-d1)
        rho = -K * T * disc_r * std_norm_cdf(-d2)
        theta = (- (S * disc_q * pdf_d1 * sigma) / (2 * math.sqrt(T))
                 + r * K * disc_r * std_norm_cdf(-d2)
                 - q * S * disc_q * std_norm_cdf(-d1))

    gamma = disc_q * pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * disc_q * pdf_d1 * math.sqrt(T)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }

def implied_vol_bisect(target_price: float, S: float, K: float, r: float, q: float, T: float, opt: OptionType,
                       tol: float = 1e-6, max_iter: int = 100) -> float:
    """Find implied volatility by bisection on [1e-6, 5.0]."""
    low, high = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        price = bs_price(S, K, r, q, mid, T, opt)
        if abs(price - target_price) < tol:
            return mid
        # if model price too low => need higher sigma
        if price < target_price:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)

# --- Cox–Ross–Rubinstein (binomial) ---

def crr_binomial_price(S: float, K: float, r: float, q: float, sigma: float, T: float, N: int, opt: OptionType) -> float:
    """CRR binomial tree for a European option."""
    dt = T / N
    if dt <= 0:
        return max(0.0, S - K) if opt == "C" else max(0.0, K - S)
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = max(0.0, min(1.0, p))
    # terminal payoffs
    prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]
    if opt == "C":
        values = [max(0.0, x - K) for x in prices]
    else:
        values = [max(0.0, K - x) for x in prices]
    # backward induction
    for _ in range(N):
        values = [disc * (p * values[j + 1] + (1 - p) * values[j]) for j in range(len(values) - 1)]
    return float(values[0])

# --- Monte Carlo (risk-neutral) ---

def mc_price(S: float, K: float, r: float, q: float, sigma: float, T: float, n_paths: int, opt: OptionType,
             seed: Optional[int] = 42) -> float:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    drift = (r - q - 0.5 * sigma**2) * T
    diff = sigma * math.sqrt(T) * Z
    ST = S * np.exp(drift + diff)
    if opt == "C":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    return float(math.exp(-r * T) * np.mean(payoff))
