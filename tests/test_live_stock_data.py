import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app_module = importlib.import_module('app')


def test_get_stock_data_uses_fallback_when_yfinance_fails(monkeypatch):
    def fake_ticker(_ticker):
        raise RuntimeError('boom')

    monkeypatch.setattr(app_module.yf, 'Ticker', fake_ticker)
    result = app_module.get_stock_data('AAPL')

    assert result['ticker'] == 'AAPL'
    assert result['company_name']
    assert result['current_price'] > 0
