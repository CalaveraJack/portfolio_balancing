from __future__ import annotations

import numpy as np


def fmt_pct(x: object) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return str(x)


def fmt_num(x: object) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def fmt_bp(x: object) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x) * 100:.1f} bp"
    except Exception:
        return str(x)
