# debug_caps.py
import pandas as pd
from pathlib import Path

caps_path = Path("data/market_caps.parquet")
if caps_path.exists():
    caps = pd.read_parquet(caps_path)
    print(f"Caps shape: {caps.shape}")
    print(f"Caps columns: {caps.columns.tolist()}")
    print(f"Caps index: {caps.index[:5]} to {caps.index[-5:]}")
    print(f"\nFirst row of caps:")
    print(caps.iloc[0])
    print(f"\nLast row of caps:")
    print(caps.iloc[-1])
    print(f"\nAre any values non-zero?")
    print(f"Non-zero count: {(caps > 0).sum().sum()}")
else:
    print("No caps file found")