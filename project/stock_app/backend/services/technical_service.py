import yfinance as yf
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator


async def get_technical_info(ticker: str):
    """
    yfinance, ta api 이용
    """

    # 주가 데이터 다운로드
    df = yf.download(ticker, period="1y", interval="1d")

    # 종가
    # df["Close"] # 행렬 기준으로 가져옴 -> 시리즈 구조로 바꿔줌
    close = df["Close"].squeeze()
    if close.empty or len(close) < 2:
        return {
            "current_price": None,
            "sma20": None,
            "sma60": None,
            "rsi": None,
            "macd": None,
            "macd_signal": None,
            "trend": "unknown",
        }

    macd_value = None
    signal_value = None
    trend = "unknown"
    try:
        macd = MACD(close)
        macd_value = float(macd.macd().iloc[-1])
        signal_value = float(macd.macd_signal().iloc[-1])
        trend = "bullish" if macd_value > signal_value else "bearish"
    except Exception:
        pass

    return {
        # 현재주가
        "current_price": float(close.iloc[-1]),
        # 20일선
        "sma20": float(SMAIndicator(close, window=20).sma_indicator().iloc[-1])
        if len(close) >= 20
        else None,
        # 60일선
        "sma60": float(SMAIndicator(close, window=60).sma_indicator().iloc[-1])
        if len(close) >= 60
        else None,
        # rsi
        "rsi": float(RSIIndicator(close, window=14).rsi().iloc[-1])
        if len(close) >= 14
        else None,
        # macd
        "macd": macd_value,
        # macd_signal
        "macd_signal": signal_value,
        # trend
        "trend": trend,
    }
