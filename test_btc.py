import pytest
import yfinance as yf


def test_btc_price_data_is_available_when_yahoo_returns_data():
    try:
        price = yf.Ticker('BTC-USD').history(period='1d')['Close'].iloc[-1]
    except Exception as exc:
        pytest.skip(f'BTC data unavailable from Yahoo Finance: {exc}')

    assert price > 0
