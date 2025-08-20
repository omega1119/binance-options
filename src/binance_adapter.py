
import os, json, time
from typing import Optional, Dict, Any, List

import requests

BASE = "https://eapi.binance.com"

def _get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10):
    url = f"{BASE}{path}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def exchange_info() -> Dict[str, Any]:
    """GET /eapi/v1/exchangeInfo"""
    return _get("/eapi/v1/exchangeInfo")

def mark(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """GET /eapi/v1/mark"""
    params = {"symbol": symbol} if symbol else None
    return _get("/eapi/v1/mark", params=params)

def index_price(underlying: str) -> Dict[str, Any]:
    """GET /eapi/v1/index for underlying (e.g., BTCUSDT)"""
    return _get("/eapi/v1/index", params={"underlying": underlying})

def load_sample_mark_data(sample_path: str) -> List[Dict[str, Any]]:
    with open(sample_path, "r") as f:
        return json.load(f)
