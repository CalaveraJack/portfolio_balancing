"""
Pre-load market cap data for offline use.
Run this once while you have internet access.
"""

from main import load_market_caps, PHARMA_48

if __name__ == "__main__":
    print("=" * 60)
    print("Pre-loading Market Cap Data")
    print("=" * 60)
    
    caps = load_market_caps(PHARMA_48, use_cache_only=False)
    
    if not caps.empty:
        print(f"\n✓ Successfully loaded caps for {len(caps.columns)} tickers")
        print(f"  Date range: {caps.index.min().date()} to {caps.index.max().date()}")
        print(f"  Latest caps (billions USD):")
        print(caps.iloc[-1].sort_values(ascending=False).head(10))
    else:
        print("\n✗ Failed to load market cap data")