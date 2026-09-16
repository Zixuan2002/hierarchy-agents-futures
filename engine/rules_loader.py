from __future__ import annotations
from pathlib import Path
from datetime import datetime
import configparser
from typing import Dict, Optional, Union

# -------- robust project root detection (engine/ parent) ----------
_THIS_FILE = Path(__file__).resolve()
ENGINE_DIR = _THIS_FILE.parent                   # .../<project>/engine
PROJECT_ROOT = ENGINE_DIR.parent                 # .../<project>
INI_DIR = PROJECT_ROOT / "data" / "market_rules" / "ini"

class RulesNotFoundError(Exception):
    pass

def _parse_date_from_name(p: Path) -> Optional[datetime]:
    """Extract YYYYMMDD from filename like 20251028.ini"""
    try:
        return datetime.strptime(p.stem, "%Y%m%d")
    except Exception:
        return None

def _normalize_date(date_like: Union[str, datetime]) -> datetime:
    if isinstance(date_like, datetime):
        return date_like
    s = str(date_like).strip()
    # Accept 'YYYYMMDD' or 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM[:SS]'
    try:
        if "-" in s:
            # take first 10 chars
            return datetime.strptime(s[:10], "%Y-%m-%d")
        return datetime.strptime(s, "%Y%m%d")
    except Exception as e:
        raise ValueError("date must be 'YYYYMMDD' or 'YYYY-MM-DD[ ...]' or datetime") from e

def select_ini_for(date_like: Union[str, datetime]) -> Path:
    """
    Select the ini file whose date <= given date and is the latest among them.
    If none found, raise RulesNotFoundError with absolute paths printed.
    """
    date_obj = _normalize_date(date_like)

    if not INI_DIR.exists():
        raise RulesNotFoundError(
            f"INI directory not found: {INI_DIR.resolve()}\n"
            f"(Make sure your project has data/market_rules/ini/<YYYYMMDD>.ini)"
        )

    candidates = []
    for p in INI_DIR.glob("*.ini"):
        d = _parse_date_from_name(p)
        if d and d <= date_obj:
            candidates.append((d, p))

    if not candidates:
        raise RulesNotFoundError(
            f"No ini file with date <= {date_obj.date()} under {INI_DIR.resolve()}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def load_rules(ini_path: Path) -> Dict[str, Dict]:
    """
    Parse ini into a dict of {SECTION: {key: value}}.
    Section names are converted to UPPERCASE.
    fee_method coerced to UPPERCASE (AMOUNT/RATIO).
    margin_ratio & fee_value converted to float.
    """
    cfg = configparser.ConfigParser()
    with ini_path.open("r", encoding="utf-8") as f:
        cfg.read_file(f)

    rules: Dict[str, Dict] = {}
    for sect in cfg.sections():
        up = sect.upper()
        raw = {k.strip(): v.strip() for k, v in cfg[sect].items()}

        entry: Dict[str, Union[str, float]] = {}
        if "exchange" in raw:
            entry["exchange"] = raw["exchange"].upper()
        if "margin_ratio" in raw:
            entry["margin_ratio"] = float(raw["margin_ratio"])
        if "fee_method" in raw:
            entry["fee_method"] = raw["fee_method"].upper()
        if "fee_value" in raw:
            entry["fee_value"] = float(raw["fee_value"])
        if "note" in raw:
            entry["note"] = raw["note"]

        rules[up] = entry
    return rules

def _extract_symbol(contract_code: str) -> str:
    """
    Extract leading letters as symbol; convert to UPPERCASE.
    e.g., rb2501 -> RB, IF2503 -> IF
    """
    s = contract_code.strip()
    if not s:
        raise ValueError("empty contract_code")
    i = 0
    n = len(s)
    while i < n and s[i].isalpha():
        i += 1
    if i == 0:
        raise ValueError(f"contract_code has no leading letters: {contract_code}")
    return s[:i].upper()

def query(contract_code: str, date_like: Union[str, datetime]) -> Dict:
    """
    Public API: given contract code and a date, return rule dict:
    {exchange, margin_ratio, fee_method, fee_value, note?, _source_section, _ini_path}
    Priority: [CONTRACT_CODE_UPPER] > [SYMBOL_UPPER] > [DEFAULT]
    """
    ini_path = select_ini_for(date_like)
    rules = load_rules(ini_path)

    full_key = contract_code.strip().upper()
    symbol = _extract_symbol(contract_code)

    if full_key in rules:
        out = rules[full_key].copy()
        out["_source_section"] = full_key
        out["_ini_path"] = str(ini_path.resolve())
        out["_project_root"] = str(PROJECT_ROOT.resolve())
        return out
    if symbol in rules:
        out = rules[symbol].copy()
        out["_source_section"] = symbol
        out["_ini_path"] = str(ini_path.resolve())
        out["_project_root"] = str(PROJECT_ROOT.resolve())
        return out
    if "DEFAULT" in rules:
        out = rules["DEFAULT"].copy()
        out["_source_section"] = "DEFAULT"
        out["_ini_path"] = str(ini_path.resolve())
        out["_project_root"] = str(PROJECT_ROOT.resolve())
        return out

    raise RulesNotFoundError(
        f"No matching section found for {full_key} or {symbol} in {ini_path.name}\n"
        f"Available sections: {list(rules.keys())[:10]}..."
    )

# Expose paths for debugging from notebooks
def debug_paths() -> Dict[str, str]:
    return {
        "this_file": str(_THIS_FILE),
        "engine_dir": str(ENGINE_DIR),
        "project_root": str(PROJECT_ROOT),
        "ini_dir": str(INI_DIR),
    }
